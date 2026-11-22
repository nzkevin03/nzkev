# Hello, I made some changes on 2026/11/22
# *******************************************
# %% [markdown]
# # Extract Semantic Model Sizes → Lakehouse Delta Table
# #
# Enumerate the semantic models (datasets) in the workspace inventory table, read each
# model's size in bytes from its Vertipaq storage stats over the XMLA endpoint
# (via semantic-link-labs `get_semantic_model_size`), and upsert the results into
# the Lakehouse Delta table `dbo.workspace_datasets_size`.
# #
# Because the XMLA endpoint only exists on dedicated capacity (Premium / PPU /
# Fabric), the inventory is filtered before sizing to:
# - keep only workspaces where `dbo.workspace.ws_is_on_dedicated_capacity = 1`
# (Pro workspaces, flagged `0`, are dropped — they have no XMLA endpoint); and
# - drop auto-generated models whose name contains "Usage Metrics Report"
# (see `DATASET_NAME_EXCLUDE_FRAGMENTS`), which expose no XMLA size.
# #
# Authentication uses the notebook's own running identity (the interactive user, or
# the service principal the scheduled job runs as) — no access token is fetched. Shared
# helpers (Lakehouse path builders and the SQL endpoint refresh) are loaded from
# `nb_pbi_shared_utils`.
# #
# Requirements
# - The source inventory table (`lh_fabric_audit.dbo.workspace_datasets`) and the
# workspace table (`lh_fabric_audit.dbo.workspace`) must be available.
# - The semantic models must live on Fabric / Premium / PPU capacity (XMLA read enabled)
# so model storage stats are available.
# - The notebook's running identity needs Build permission (or workspace
# Contributor/Member/Admin) on each model to read it over XMLA.
# %% [markdown]
# ## Load Shared Utilities and Imports
# %%
# %run nb_pbi_shared_utils
# %%
# Install semantic-link-labs into the current kernel, then restart Python so the
# new package is importable. In Fabric, %pip must run before any other imports.
# %pip install semantic-link-labs -q
# %%


from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import traceback

import sempy.fabric as fabric
import sempy_labs as labs
from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


REQUIRED_SYMBOLS = [
    "build_lakehouse_path",
    "build_paths",
    "path_exists",
    "refresh_lakehouse_sql_endpoint",
    "require_shared_symbols",
    "resolve_run_date",
]
if "require_shared_symbols" in globals():
    require_shared_symbols(
    globals(),
    REQUIRED_SYMBOLS,
    "extract_semantic_model_sizes",
)
else:
    missing_symbols = [name for name in REQUIRED_SYMBOLS if name not in globals()]
    raise ImportError(
        "Missing shared utilities for extract_semantic_model_sizes: "
        f"{missing_symbols}. Load nb_pbi_shared_utils via %run before running this notebook."
)

## Parameters and Configuration

WORKSPACE_ID = "22307e6a-3183-4f98-a297-701e85d91392"
LAKEHOUSE_ID = "b086dea5-777f-4692-a8db-f5f4351f1035"
SQL_ENDPOINT_ID = "e8f26588-a230-473f-97b5-5909bb47d3c3"
RUN_DATE_YYYYMMDD_PARAM = None
PROGRESS_INTERVAL = 50
REFRESH_SQL_ENDPOINT = True
# Datasets whose name contains any of these fragments are skipped (no XMLA size).
DATASET_NAME_EXCLUDE_FRAGMENTS = ["Usage Metrics Report", "Azure Synapse Monitor"]
# --- Service principal auth for the XMLA size calls ---
# When True, the sizes are read as the 'PowerBI-Data-Automation' service principal
# using TENANT_ID / CLIENT_ID / CLIENT_SECRET, which nb_pbi_shared_utils loads from
# Key Vault in its "Get Auth token" cell. When False, the notebook's own identity
# is used.
USE_SERVICE_PRINCIPAL = True
TZ = "Pacific/Auckland"
LH_PATH = build_lakehouse_path(WORKSPACE_ID, LAKEHOUSE_ID)
PATHS = build_paths(LH_PATH)
CONFIG = {
    # Read via the ABFSS Delta path so no default Lakehouse needs to be attached.
    "DATASET_SOURCE_PATH": f"{PATHS['tables_base']}/dbo/workspace_datasets",
    # Workspace table carries ws_is_on_dedicated_capacity (1 = Premium/PPU/Fabric).
    "WORKSPACE_SOURCE_PATH": f"{PATHS['tables_base']}/dbo/workspace",
    "SIZE_TABLE_PATH": f"{PATHS['tables_base']}/dbo/workspace_datasets_size",
}


## Output Schema

SIZE_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("run_date", StringType(), False),
        StructField("workspace_id", StringType(), False),
        StructField("workspace_name", StringType(), True),
        StructField("dataset_id", StringType(), False),
        StructField("dataset_name", StringType(), True),
        StructField("size_bytes", LongType(), True),
        StructField("size_mb", DoubleType(), True),
        StructField("size_gb", DoubleType(), True),
        StructField("collection_status", StringType(), False),
        StructField("error_message", StringType(), True),
        StructField("observed_at_utc", TimestampType(), False),
        StructField("observed_at_nz", TimestampType(), False),
    ]
)

def _to_nz_timestamp(utc_timestamp: Optional[datetime]) -> Optional[datetime]:
    if utc_timestamp is None:
        return None
    return (
        utc_timestamp.replace(tzinfo=timezone.utc)
        .astimezone(ZoneInfo(TZ))
        .replace(tzinfo=None)
    )

def _load_dedicated_workspace_ids() -> set:
    """Workspace ids on dedicated capacity (Premium/PPU/Fabric) with XMLA support.
    The dbo.workspace table flags Pro workspaces with
    ws_is_on_dedicated_capacity = 0; those have no XMLA endpoint and are excluded.
    """
    source_path = CONFIG["WORKSPACE_SOURCE_PATH"]
    if not path_exists(source_path):
        raise ValueError(
            f"Workspace table not found: {source_path}. "
            "Confirm it holds dbo.workspace with ws_id and "
            "ws_is_on_dedicated_capacity columns."
    )
    workspace_df = spark.read.format("delta").load(source_path)
    
    id_column = next(
        (name for name in ("ws_id", "workspace_id", "id") if name in workspace_df.columns),
        None,
    )
    if id_column is None or "ws_is_on_dedicated_capacity" not in workspace_df.columns:
        raise ValueError(
            f"{source_path} missing required columns. Need a workspace id column "
            "(ws_id/workspace_id/id) and ws_is_on_dedicated_capacity. "
            f"Actual columns: {workspace_df.columns}"
    )
    dedicated_flag = F.lower(F.trim(F.col("ws_is_on_dedicated_capacity").cast("string")))
    rows = (
        workspace_df.select(
            F.col(id_column).cast("string").alias("workspace_id"),
            dedicated_flag.alias("is_dedicated"),
        )
        .where(
            F.col("workspace_id").isNotNull()
            & (F.trim(F.col("workspace_id")) != "")
            & F.col("is_dedicated").isin("1", "true")
        )
        .select("workspace_id")
        .distinct()
        .collect()
    )
    return {row["workspace_id"] for row in rows}

def _load_inventory_rows() -> List[Dict[str, Optional[str]]]:
    source_path = CONFIG["DATASET_SOURCE_PATH"]
    if not path_exists(source_path):
        raise ValueError(
            f"Semantic model inventory path not found: {source_path}. "
            "Confirm WORKSPACE_ID/LAKEHOUSE_ID point at the audit Lakehouse "
            "that holds dbo.workspace_datasets."
        )
    source_df = spark.read.format("delta").load(source_path)
    required_columns = ["workspace_id", "workspace_name", "datasets_id", "datasets_name"]
    missing_columns = [name for name in required_columns if name not in source_df.columns]
    if missing_columns:
        raise ValueError(
            f"{source_path} missing required columns {missing_columns}. "
            f"Actual columns: {source_df.columns}"
    )
    # Keep only workspaces on dedicated capacity (Pro workspaces have no XMLA endpoint).
    dedicated_workspace_ids = _load_dedicated_workspace_ids()
    base_df = source_df.select(
        F.col("workspace_id").cast("string").alias("workspace_id"),
        F.col("workspace_name").cast("string").alias("workspace_name"),
        F.col("datasets_id").cast("string").alias("dataset_id"),
        F.col("datasets_name").cast("string").alias("dataset_name"),
    ).where(
        F.col("workspace_id").isNotNull()
        & (F.trim(F.col("workspace_id")) != "")
        & F.col("dataset_id").isNotNull()
        & (F.trim(F.col("dataset_id")) != "")
        & F.col("workspace_id").isin(list(dedicated_workspace_ids))
    )
    # Exclude auto-generated models that have no XMLA size (e.g. Usage Metrics Report).
    for fragment in DATASET_NAME_EXCLUDE_FRAGMENTS:
        base_df = base_df.where(
        F.col("dataset_name").isNull()
        | (~F.lower(F.col("dataset_name")).contains(fragment.lower()))
    )
    rows = base_df.dropDuplicates(["workspace_id", "dataset_id"]).collect()
    return sorted(
        (row.asDict(recursive=True) for row in rows),
        key=lambda row: (row["workspace_id"], row["dataset_id"]),
    )

def _xmla_auth_context():
    # Authenticate every sempy call in the block (including the XMLA-based
    # get_semantic_model_size) as the 'PowerBI-Data-Automation' service principal.
    # TENANT_ID / CLIENT_ID / CLIENT_SECRET are loaded from Key Vault by
    # nb_pbi_shared_utils. When USE_SERVICE_PRINCIPAL is False, the notebook's own
    # running identity is used (no-op context).
    if not USE_SERVICE_PRINCIPAL:
        return nullcontext()
        
    missing = [
        name
        for name in ("TENANT_ID", "CLIENT_ID", "CLIENT_SECRET")
        if not globals().get(name)
    ]
    if missing:
        raise ValueError(
            f"Service principal credentials {missing} not found. Run "
            "nb_pbi_shared_utils (the 'Get Auth token' cell) before this notebook, "
            "or set USE_SERVICE_PRINCIPAL = False."
        )
    return fabric.set_service_principal(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

def _get_model_size_bytes(dataset_id: str, workspace_id: str) -> Optional[int]:
    # Connects to the XMLA endpoint using whichever identity the active auth
    # context provides (service principal or notebook user).
    return labs.get_semantic_model_size(dataset=dataset_id, workspace=workspace_id)

def _merge_size_delta(source_df: DataFrame, target_path: str) -> None:
    merge_condition = """
        target.workspace_id = source.workspace_id
        AND target.dataset_id = source.dataset_id
    """
    key_columns = {"workspace_id", "dataset_id"}
    update_assignments = {
        column_name: f"source.`{column_name}`"
        for column_name in source_df.columns
        if column_name not in key_columns
    }
    if DeltaTable.isDeltaTable(spark, target_path):
        (
            DeltaTable.forPath(spark, target_path)
            .alias("target")
        .merge(source_df.alias("source"), merge_condition)
        .whenMatchedUpdate(set=update_assignments)
        .whenNotMatchedInsertAll()
        .execute()
        )
    else:
        (
        source_df.write.format("delta")
        .mode("overwrite")
        .option("delta.enableChangeDataFeed", "true")
        .save(target_path)
        )

 
## Collect, Normalize, and Write Semantic Model Sizes
 
run_date = resolve_run_date(RUN_DATE_YYYYMMDD_PARAM, TZ)
run_id = datetime.now(ZoneInfo(TZ)).strftime("%Y%m%d_%H%M%S")
observed_at = datetime.now(timezone.utc).replace(tzinfo=None)
observed_at_nz = _to_nz_timestamp(observed_at)

inventory_rows = _load_inventory_rows()
if not inventory_rows:
    raise RuntimeError(
        f"No semantic model inventory rows found in {CONFIG['DATASET_SOURCE_PATH']}."
    )
    
size_rows: List[Dict[str, Any]] = []
successful_models = 0
failed_models = 0

auth_mode = "service_principal" if USE_SERVICE_PRINCIPAL else "notebook_identity"
print(f"run_id={run_id}")
print(f"run_date={run_date}")
print(f"auth_mode={auth_mode}")
print(f"models_to_size={len(inventory_rows)}")

with _xmla_auth_context():
    for model_index, artifact in enumerate(inventory_rows, start=1):
        try:
            raw_size_bytes = _get_model_size_bytes(
                artifact["dataset_id"],
                artifact["workspace_id"],
            )
            size_bytes = int(raw_size_bytes) if raw_size_bytes is not None else None
            collection_status = "success"
            error_message = None
            successful_models += 1
        except Exception as error: # models without XMLA read access fail here
            size_bytes = None
            collection_status = "error"
            error_message = (
                f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            )[:4000]
            failed_models += 1
        
        size_rows.append(
            {
                "run_id": run_id,
                "run_date": run_date,
                "workspace_id": artifact["workspace_id"],
                "workspace_name": artifact.get("workspace_name"),
                "dataset_id": artifact["dataset_id"],
                "dataset_name": artifact.get("dataset_name"),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / 1024 / 1024, 2) if size_bytes is not None else None,
                "size_gb": round(size_bytes / 1024 / 1024 / 1024, 4) if size_bytes is not None else None,
                "collection_status": collection_status,
                "error_message": error_message,
                "observed_at_utc": observed_at,
                "observed_at_nz": observed_at_nz,
            }
        )
        
        if model_index % PROGRESS_INTERVAL == 0:
            print(f"processed={model_index}/{len(inventory_rows)}")
        
print(
    f"completed: models={len(inventory_rows)}, successful={successful_models}, "
    f"failed={failed_models}"
)



size_df = spark.createDataFrame(size_rows, SIZE_SCHEMA).dropDuplicates(
    ["workspace_id", "dataset_id"]
)
_merge_size_delta(size_df, CONFIG["SIZE_TABLE_PATH"])
print(f"merged_rows={size_df.count()} into {CONFIG['SIZE_TABLE_PATH']}")

## Refresh Lakehouse SQL Endpoint

run_failures: List[str] = []
if successful_models == 0:
    run_failures.append("Semantic model size collection failed for every inventory model.")
if REFRESH_SQL_ENDPOINT:
    # The SQL endpoint reports schema-enabled Lakehouse tables as "schema.table".
    size_table_name = CONFIG["SIZE_TABLE_PATH"].rstrip("/").split("/")[-1]
    sql_endpoint_table = f"dbo.{size_table_name}"
    try:
        refresh_result = refresh_lakehouse_sql_endpoint(
            workspace_id=WORKSPACE_ID,
            sql_endpoint_id=SQL_ENDPOINT_ID,
            timeout_minutes=5,
            required_tables=[sql_endpoint_table],
        )
        print(
            f"sql_endpoint_refresh_status={refresh_result.get('status')}, "
            f"message={refresh_result.get('message')}, "
            f"required_tables={refresh_result.get('required_table_statuses')}, "
            f"failed_tables={refresh_result.get('failed_tables')}"
        )
    except Exception as error:
        run_failures.append(
            "SQL endpoint refresh failed: "
            f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        )
if run_failures:
    raise RuntimeError("; ".join(run_failures))