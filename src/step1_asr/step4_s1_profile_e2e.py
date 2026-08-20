import qai_hub as hub
import json
import os

def main():
    # Load compile job ids to get the compiled model ID
    job_ids_path = "outputs/sensevoice-e2e-onnx/e2e_qai_job_ids.json"
    if not os.path.exists(job_ids_path):
        print(f"Error: Could not find {job_ids_path}")
        return

    with open(job_ids_path, "r", encoding="utf-8") as f:
        job_ids = json.load(f)
        
    compiled_model_id = job_ids["compiled_model_id"]
    print(f"Loaded compiled model ID: {compiled_model_id}")
    
    device = hub.Device("Dragonwing IQ-9075 EVK")
    compiled_model = hub.get_model(compiled_model_id)
    
    print("Submitting profile job...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="SenseVoice_E2E_Profile",
    )
    
    print(f"Profile job submitted: {profile_job.job_id}")
    print(f"URL: {profile_job.url}")
    
    print("Waiting for completion...")
    profile_job.wait()
    
    print("\n--- Profile Results ---")
    data = profile_job.download_profile()
    
    print("Execution Summary:")
    print(data.execution_summary)
    
if __name__ == "__main__":
    main()
