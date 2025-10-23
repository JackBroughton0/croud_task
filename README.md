# Sentiment Analyzer
A simple implementation of a sentiment analyzer class with unit tests that run via a GitHub Actions workflow.

---

## Features

- **AWS Comprehend** for fast, language-aware sentiment detection  
- **OpenAI (optional)** for extra analysis to boost confidence in sentiment classification if desired  
- **Emoji weighting system** to reflect tone in informal text  
- **Configurable neutrality band** for control of neutral classification

---

### How to run
1) Set up a virtual environment using UV.

   ```
   uv venv
   source .venv/bin/activate
   ```

2) Install the dependencies

   ```
   uv sync --dev
   ```

3) Environment variables: create a ```.env``` file in the ```src/``` directory with the following:

    ```
    AWS_ACCESS_KEY_ID=your_aws_access_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret_key
    AWS_REGION=eu-west-2
    OPENAI_API_KEY=your_openai_key
    OPENAI_MODEL=gpt-4o-mini
    ```

---

### Usage
Run the analyzer directly:
```
uv run python sentiment_analyzer/base.py
```
**Note that more example comments can be added within the base.py file. There are only a handful of existing comments to run through here to keep it simple and due to cost implications.**

---

### Tests
The project includes unit tests written with pytest. Run all tests with:
```
uv run pytest -v
```
CI/CD tests are automatically triggered on push and pull requests via GitHub Actions (see ```.github/workflows/sentiment_analyzer_ci.yml```).

---

### Future Improvements

1) **Modularity and Composition**

    Currently, AWS Comprehend and OpenAI logic live inside base.py.
    To improve scalability and testability, we could refactor each provider into its own module:
    - comprehend_client.py for AWS Comprehend
    - openai_client.py for OpenAI
    - base.py remains the orchestration layer (composition-based approach)

2) **Secure Secrets Management**

    - Replace ```.env``` usage with AWS Secrets Manager
    - Use IAM roles for AWS Comprehend authentication (no hardcoded credentials)

3) **Reliability and Retries**

    Add exponential backoff retry mechanisms for:
    - AWS Comprehend API calls (botocore can be configured)
    - OpenAI API responses (use HTTP retry logic or async wrapper)

4) **Containerization with Docker**

5) **Infrastructure as Code (IaC) with Terraform**

---

### AI Assistance Disclaimer
All ideas, solutions, and implementation decisions in this submission are my own. I used ChatGPT and GitHub Copilot for guidance, but any AI-generated code snippets were specifically prompted, reviewed, tested, and modified by me to ensure accuracy and alignment with my own understanding.