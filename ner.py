
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

text = """
The patient was diagnosed with pneumonia.
The patient also has diabetes.
"""

results = ner(text)

text = """
Patient is a 45-year-old female with a history of
type 2 diabetes mellitus and hypertension. She presents
with complaints of shortness of breath and cough. On
examination, she has a temperature of 102.4°F,
heart rate 110, blood pressure 160/100, and oxygen
saturation 92% on room air.
Chest X-ray shows lobar consolidation in the right
lower lobe. Sputum Gram stain is positive for
Streptococcus pneumoniae. She is started on IV
ceftriaxone and azithromycin.
"""

results = ner(text)

print("Raw model output:")
print(results)
print("Number of entities:", len(results))
