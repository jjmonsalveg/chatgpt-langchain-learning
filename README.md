# Chatgpt and Langchain

## Prerequisites

1. If you have not already done so, create a pycode directory somewhere on your development machine.
2. In your terminal run pip install pipenv or depending on your environment, 

```shell
pyenv install 3.11.10
pyenv shell 3.11.10
python -V
pip install pipenv
```

3. Create a file in your pycode project directory called Pipfile
4. Copy paste the following code into that new Pipfile (or drag and drop the
file that is attached to this lecture into your pycode project directory)

```Pipfile
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"
 
[packages]
langchain = "==0.0.352"
openai = "==0.27.8"
python-dotenv = "==1.0.0"
 
[dev-packages]
 
[requires]
python_version = "3.11"
```

5. Inside your pycode project directory, run the following command to install your dependencies from the Pipfile:

```shell
pipenv install
pipenv --venv # to verify the environment created
```

6. Run the following command to create and enter a new environment:

```shell
pipenv shell
```

After doing this your terminal will now be running commands in this new environment managed by Pipenv.

Once inside this shell, you can run Python commands just as shown in the lecture videos.

eg:

python main.py

7. If you make any changes to your environment variables or keys, you may find
that you need to exit the shell and re-enter using the pipenv shell command.

## Obtaining API Key

1. Open your browser and navigate to https://platform.openai.com/

2. Create an OpenAI account by logging in with an existing provider or by creating an account with your email address.

3. In the top menu click on `Dashboard` and then in the left menu click on `Api Keys`

4. Click in `Create new secret key`

5. put a name in this case `PyCode`, and copy the key generated since this key will be lost and not shown again.

## Links

- [Discord comunnity](https://discord.gg/h2G3CbxPZA)
