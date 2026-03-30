from transformers import T5ForConditionalGeneration, RobertaTokenizer
import torch


class HFClient:
    def __init__(self, model_name="ankit-ml11/automerge-codet5"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt, max_length=512, num_beams=5):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def resolve_conflict(self, base, ours, theirs, language="python"):
        """Resolve a merge conflict given base, ours, and theirs versions."""
        input_text = f"""Resolve the following merge conflict in {language}.

BASE VERSION:
{base}

OURS VERSION:
{ours}

THEIRS VERSION:
{theirs}
"""
        return self.generate(input_text)

    @staticmethod
    def parse_git_conflict(conflict_text):
        """Parse standard Git conflict markers into base, ours, and theirs."""
        lines = conflict_text.split('\n')
        ours, base, theirs = [], [], []
        section = None

        for line in lines:
            if line.startswith('<<<<<<<'):
                section = 'ours'
            elif line.startswith('|||||||'):
                section = 'base'
            elif line.startswith('======='):
                section = 'theirs'
            elif line.startswith('>>>>>>>'):
                section = None
            elif section == 'ours':
                ours.append(line)
            elif section == 'base':
                base.append(line)
            elif section == 'theirs':
                theirs.append(line)

        return {
            'base': '\n'.join(base) or '\n'.join(ours),
            'ours': '\n'.join(ours),
            'theirs': '\n'.join(theirs)
        }