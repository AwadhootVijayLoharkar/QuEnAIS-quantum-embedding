from quenais.workflows.dft_pipeline_pipeline import run_formaldehyde_test

if __name__ == "__main__":
    results = run_formaldehyde_test()
    print("DFT Energy:", results["energy"])