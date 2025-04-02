def summarize(df):
    print(f"Shape: {df.shape}")

    print(f"Number of missing values: {df.isna().sum().sum()}")

    display(df.head())