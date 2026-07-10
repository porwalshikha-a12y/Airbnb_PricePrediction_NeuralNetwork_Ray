from airbnb_azure_nn import run_full_pipeline

def run():
    print("Starting Airbnb pipeline...")
    df = run_full_pipeline()
    print("Pipeline finished.")
    return df

if __name__ == "__main__":
    run()
