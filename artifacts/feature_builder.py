import pandas as pd


def build_features(users, visits, ads_activity, surf_depth, primary_device, cloud_usage):
    """
    Собирает единый датасет и формирует признаки так же,
    как это было сделано при обучении модели.
    """

    users = users.drop_duplicates()
    visits = visits.drop_duplicates()
    ads_activity = ads_activity.drop_duplicates(subset="user_id")

    # Агрегация логов
    visits_by_category = visits.pivot_table(
        index="user_id",
        columns="website_category",
        values="session_id",
        aggfunc="count",
        fill_value=0
    )

    visits_by_daytime = visits.pivot_table(
        index="user_id",
        columns="daytime",
        values="session_id",
        aggfunc="count",
        fill_value=0
    )

    user_activity = visits.groupby("user_id").agg(
        total_visits=("session_id", "count"),
        unique_days_count=("date", "nunique")
    )

    visits_features = user_activity.join([visits_by_category, visits_by_daytime])

    # Объединение таблиц
    data = (
        users.set_index("user_id")
        .join([
            visits_features,
            ads_activity.set_index("user_id"),
            surf_depth.set_index("user_id"),
            primary_device.set_index("user_id"),
            cloud_usage.set_index("user_id"),
        ])
    )

    # Удаление неинформативного признака
    if "unique_days_count" in data.columns:
        data = data.drop(columns=["unique_days_count"])

    # Генерация долей
    category_columns = data.filter(like="Category").columns.tolist()
    time_columns = ["утро", "день", "вечер", "ночь"]
    columns_to_normalize = category_columns + time_columns

    for column in columns_to_normalize:
        data[f"share_{column}"] = data[column] / data["total_visits"]

    data = data.drop(columns=columns_to_normalize)
    data = data.drop(columns=["total_visits"])

    # Приведение типов
    data["cloud_usage"] = data["cloud_usage"].replace({True: "True", False: "False"})
    data["cloud_usage"] = data["cloud_usage"].astype("object")

    return data
