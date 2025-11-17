from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from sklearn.ensemble import (
    HistGradientBoostingRegressor as HGBR,
    HistGradientBoostingClassifier as HGBC,
)
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
    GridSearchCV,
)


# Define constants for paths
DOWNSTREAM_PATH = Path(__file__).parents[1] / "data" / "downstream_tables"
WDBS_CLS_PATH = DOWNSTREAM_PATH / "wikidbs/classification"
WDBS_REG_PATH = DOWNSTREAM_PATH / "wikidbs/regression"
RW_PATH = DOWNSTREAM_PATH / "real_world"

VALIDATION_TABLES = [
    WDBS_CLS_PATH / "73376_HISTORICAL_FIGURES.parquet",
    WDBS_CLS_PATH / "9510_CreativeCommonsAuthors.parquet",
    WDBS_REG_PATH / "62826_HISTORICAL_FIGURES.parquet",
    WDBS_REG_PATH / "66610_geopolitical_regions.parquet",
]


def prediction_scores(
    embeddings,
    embeddings_name,
    source_kg,
    entity_to_idx,
    downstream_file,
    task,
    n_repeats=5,
    tune_hyperparameters=True,
    verbose=False,
):
    """Evaluates entity embeddings on a downstream machine learning task.

    This function takes a set of pre-trained entity embeddings and evaluates
    their performance on a specified downstream task, which can be either
    classification or regression. It loads a target dataset, maps entities to
    their embeddings, and then performs repeated k-fold cross-validation using
    a HistGradientBoosting model. Hyperparameter tuning can be optionally
    performed. The results of the evaluation are saved to a json file.

    Args:
        embeddings (np.ndarray): A 2D numpy array where each row is an entity
            embedding.
        embeddings_name (str): The name of the embeddings model, used for
            saving results.
        source_kg (str): The name of the source knowledge graph (e.g.,
            "yago4.5", "freebase"). This determines which entity column to use
            from the downstream file.
        entity_to_idx (dict): A dictionary mapping entity names (str) to their
            corresponding index (int) in the `embeddings` array.
        downstream_file (str or Path): The path to the Parquet file containing
            the downstream task data. This file must contain a 'target' column
            and an entity column corresponding to the `source_kg`.
        task (str): The type of the downstream task. Must be either
            "classification" or "regression".
        n_repeats (int, optional): The number of times to repeat the k-fold
            cross-validation. Defaults to 5.
        tune_hyperparameters (bool, optional): If True, performs hyperparameter
            tuning using GridSearchCV. Defaults to True.
        verbose (bool, optional): If True, prints the evaluation summary to the
            console. Defaults to False.

    Raises:
        ValueError: If the `source_kg` is not recognized.

    Returns:
        None: The function saves the results to a json file in the
            '../results/' directory and does not return any value.
    """
    # Load target dataframe
    target = pd.read_parquet(downstream_file)

    # Replace entity names by their embedding
    nan_array = np.empty(shape=embeddings[0].shape, dtype="float32")
    nan_array[:] = np.nan

    if "yago4.5" in source_kg:
        column_to_embed = "yago4.5_entity"
    elif "yago3" in source_kg:
        column_to_embed = "yago3_entity"
    elif "yago4" in source_kg:
        column_to_embed = "yago4_entity"
    elif "freebase" in source_kg:
        column_to_embed = "freebase_entity"
    elif "wikikg" in source_kg:
        column_to_embed = "wikidata_entity"
    else:
        raise ValueError(f"Unknown source KG: {source_kg}")

    X_emb = np.vstack(
        target[column_to_embed]
        .map(entity_to_idx)
        .apply(lambda i: embeddings[int(i)] if i == i else nan_array)
        .to_numpy()
    )
    y = target["target"]

    if task == "classification":
        model = HGBC()
        scoring = "f1_weighted"
    elif task == "regression":
        model = HGBR()
        scoring = "r2"
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats)

    if tune_hyperparameters:
        param_grid = {
            "max_depth": [2, 4, 6, None],
            "min_samples_leaf": [4, 6, 10, 20],
        }
        model = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=scoring,
            cv=3,
        )
    start_time = time()
    cv_scores = cross_val_score(model, X_emb, y, cv=cv, scoring=scoring, n_jobs=15)
    duration = time() - start_time

    X_shape = X_emb.shape

    if verbose:
        print(
            f"Downstream task: {downstream_file.stem} | Task: {task} | "
            f"Scores: {cv_scores} | Mean {scoring}: {cv_scores.mean():.4f} | "
            f"Std: {cv_scores.std():.4f} | Duration: {duration:.2f} seconds"
        )

    # Save results to a dataframe
    results = {
        "embeddings_name": embeddings_name,
        "source_kg": source_kg,
        "downstream_file": str(downstream_file),
        "task": task,
        "scoring": scoring,
        "duration": duration,
        "n_samples": X_shape[0],
        "n_features": X_shape[1],
        "scores": cv_scores,
        "task": task,
    }
    df_res = pd.DataFrame([results])
    results_dir = Path(__file__).parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{downstream_file.stem}-{embeddings_name}.json"
    df_res.to_json(results_file, orient="records", lines=True)
    return results


def evaluate_test_set(embeddings, embeddings_name, source_kg, entity_to_idx):
    """Evaluates the embeddings on the test set of downstream tasks.

    Args:
        embeddings (np.ndarray): The entity embeddings matrix to be evaluated.
        embeddings_name (str): A name for the set of embeddings, used for result file naming.
        source_kg (str): The name of the source knowledge graph, used to identify the entities.
        entity_to_idx (Dict[str, int]): A dictionary mapping entity identifiers to their
            corresponding index in the `embeddings` matrix.

    Returns:
        None: This function does not return a value. It triggers evaluations that
              may print scores or save results to files.
    """
    datasets_to_process = [
        (RW_PATH, "regression"),
        (WDBS_REG_PATH, "regression"),
        (WDBS_CLS_PATH, "classification"),
    ]

    for data_path, task in datasets_to_process:
        for downstream_file in data_path.glob("*.parquet"):
            if downstream_file in VALIDATION_TABLES:
                continue
            print(f"Evaluating on: {downstream_file.stem}")
            prediction_scores(
                embeddings,
                embeddings_name,
                source_kg,
                entity_to_idx,
                downstream_file,
                task=task,
            )
    return


def evaluate_validation_set(embeddings, embeddings_name, source_kg, entity_to_idx):
    """Evaluates a set of embeddings on all validation downstream tasks.

    Args:
        embeddings (np.ndarray): The entity embeddings matrix to be evaluated.
        embeddings_name (str): A name for the set of embeddings, used for result file naming.
        source_kg (str): The name of the source knowledge graph, used to identify the entities.
        entity_to_idx (Dict[str, int]): A dictionary mapping entity identifiers to their
            corresponding index in the `embeddings` matrix.

    Returns:
        None: This function does not return a value. It triggers evaluations that
              may print scores or save results to files.
    """
    for downstream_file in VALIDATION_TABLES:
        if "classification" in downstream_file.parts[-2]:
            task = "classification"
        else:
            task = "regression"
        prediction_scores(
            embeddings,
            embeddings_name,
            source_kg,
            entity_to_idx,
            downstream_file,
            task=task,
            n_repeats=10,
            tune_hyperparameters=False,
        )
    return
