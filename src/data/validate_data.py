
from load_data import load_all_tables


def inspect_dataset():
    tables = load_all_tables()

    print("=" * 60)
    print("SMARTSHIP - DATA VALIDATION REPORT")
    print("=" * 60)

    for name, df in tables.items():
        print(f"\nTable: {name}")
        print("-" * 40)
        print(f"Shape: {df.shape}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(f"Missing values: {df.isna().sum().sum()}")
        print("\nColumn types:")
        print(df.dtypes)
        print("\nMissing values per column:")
        print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    inspect_dataset()
