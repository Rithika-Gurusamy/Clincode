
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)


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


for entity in results:
    start = entity.get("start")
    end = entity.get("end")

    original_text = (
        text[start:end]
        if start is not None and end is not None
        else entity.get("word", "")
    )

    print(f"Original evidence : {original_text!r}")
    print(f"Model prediction  : {entity.get('word')!r}")
    print(f"Entity label      : {entity.get('entity_group')}")
    print(f"Model score       : {float(entity.get('score', 0)):.3f}")
    print("-" * 45)
