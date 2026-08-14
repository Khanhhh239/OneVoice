import qai_hub as hub
import json
import os

def main():
    # Load compile job ids to get the compiled model ID
    job_ids_path = "outputs/sensevoice-onnx/qai_job_ids.json"
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
        name="SenseVoice_w8a16_Profile",
    )
    
    print(f"Profile job submitted: {profile_job.job_id}")
    print(f"URL: {profile_job.url}")
    
    print("Waiting for completion...")
    profile_job.wait()
    
    print("\n--- Profile Results ---")
    data = profile_job.download_profile()
    
    # Typically, data is a list of run details. We will print the summary
    print("Execution Summary:")
    print(data.execution_summary)
    
if __name__ == "__main__":
    main()
