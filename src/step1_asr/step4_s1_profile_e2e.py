import sys
sys.stdout.reconfigure(encoding='utf-8')
import qai_hub as hub
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOB_IDS_PATH = os.path.join(ROOT, 'outputs', 'sensevoice-e2e-onnx', 'e2e_qai_job_ids.json')

def main():
    if not os.path.exists(JOB_IDS_PATH):
        print(f"Error: Could not find {JOB_IDS_PATH}")
        return

    with open(JOB_IDS_PATH, 'r', encoding='utf-8') as f:
        job_ids = json.load(f)
        
    compiled_model_id = job_ids['compiled_model_id']
    print(f"Loaded compiled model ID: {compiled_model_id}")
    
    device = hub.Device("Dragonwing IQ-9075 EVK")
    compiled_model = hub.get_model(compiled_model_id)
    
    print("Submitting profile job to Dragonwing IQ-9075 EVK...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="SenseVoice_E2E_Profile",
    )
    
    print(f"Profile job submitted: {profile_job.job_id}")
    print(f"URL: {profile_job.url}")
    
    # Save profile job ID
    job_ids['profile_job_id'] = profile_job.job_id
    job_ids['profile_job_url'] = profile_job.url
    with open(JOB_IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(job_ids, f, indent=2)
        
    print("Waiting for completion...")
    profile_job.wait()
    
    print("\n--- Profile Results ---")
    data = profile_job.download_profile()
    
    print("Execution Summary:")
    print(data.execution_summary)
    
if __name__ == '__main__':
    main()
