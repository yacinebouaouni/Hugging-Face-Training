from huggingface_hub import Repository
from transformers import AutoTokenizer, AutoModelForMaskedLM

# initiating a repository object to clone the repository dummy to the local
repo = Repository(local_dir = "/content/model", clone_from="yacineai7/dummy")

# load the model, tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")

# pulling the latest updates
repo.git_pull()

# saving model and tokenizer files to repository local directory
model.save_pretrained(repo.local_dir)
tokenizer.save_pretrained(repo.local_dir)

# staging changes, commiting and pushing
repo.git_add()
repo.git_commit('added model and tokenizer')
repo.git_push()


