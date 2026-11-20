# IssueTriage AI

Turn a short developer report into a GitHub issue draft. Load the StockHold example, select **Analyze**, review the evidence, edit the draft, and copy the Markdown or download issue.md. The app never posts to GitHub.

## Run in VS Code on Windows

Install Python 3.11 or newer. Extract the ZIP, open the folder containing app.py in VS Code, and choose **Terminal → New Terminal**. In PowerShell run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically. Leave the terminal open while using the app. Press Ctrl+C in the terminal to stop.

## Model choices

**Local classifier** works without an account, token, or network after installation. A multinomial Naive Bayes model learns word frequencies from 30 labeled examples. The title and evidence are extracted from the report; the follow-up comes from a label-specific template. Unknown vocabulary and mixed-intent reports can lead to poor labels. Edit the suggestion before sharing.

**Hugging Face** sends the report to the Hugging Face chat completions endpoint and asks an instruction model to draft all four fields. The app checks the label and verifies that the quoted evidence occurs in the original report. Review the suggested title and follow-up yourself. Provider access, quota and model availability depend on your account.

To enable cloud mode, stop the app with Ctrl+C. In the same PowerShell terminal use the following commands. The first prompts for an inference-enabled token without showing it. Store tokens outside the repository.

```powershell
$env:HF_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Hugging Face token' -AsSecureString)).Password
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Select **Hugging Face** in the app. To change the default model, set HF_MODEL to a chat model available to your inference provider before starting the app. API details: https://huggingface.co/docs/inference-providers/tasks/chat-completion

## Check the code

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py
```

The second command scores the local classifier on 30 separate English reports, ten per class, and prints every error and a confusion matrix. The supplied version classified 27/30 correctly. For example, it labeled “Can you fix the error when I save my profile?” as a question. Asking for a fix is still a bug report. These are small, hand-authored datasets; this score only describes these examples.

To evaluate the hosted model separately, run evaluate.py with --cloud. This sends 30 requests and consumes provider quota. Cloud integration tests use simulated responses; their results do not measure hosted-model accuracy.

## Share with an employer

Add app.py, triage.py, requirements.txt, training.csv, evaluation.csv, evaluate.py, test_triage.py, and this README to a GitHub repository. For a browser link, deploy it on a Python host that supports Streamlit, select app.py as the entry point, and use local classifier mode for a token-free walkthrough. Only add a public URL to your resume after opening that URL and trying the sample.
