from input_pipeline_config import InputPipelineConfig
from input_dataset import load_jsonl_folder, extract_texts
from input_tokenizer import FlanT5TokenizerPipeline
from input_embeddings import FlanT5EmbeddingBlock


def main():
    cfg = InputPipelineConfig()

    input_folder = "/home/ap2194.UNT/micro/assessments"

    rows = load_jsonl_folder(input_folder)
    texts = extract_texts(rows, text_field=cfg.text_field)

    print(f"Extracted {len(texts)} usable text samples")

    tokenizer_pipeline = FlanT5TokenizerPipeline(
        model_name=cfg.model_name,
        max_length=cfg.max_length
    )

    dataloader = tokenizer_pipeline.build_dataloader(
        texts=texts,
        batch_size=cfg.batch_size
    )

    embedding_block = FlanT5EmbeddingBlock(
        model_name=cfg.model_name,
        device=cfg.device
    )

    for batch_idx, batch in enumerate(dataloader):
        prepared = embedding_block.prepare_batch(batch)

        print(f"\n=== Batch {batch_idx} ===")
        print("input_ids shape     :", prepared["input_ids"].shape)
        print("attention_mask shape:", prepared["attention_mask"].shape)
        print("x_emb shape         :", prepared["x_emb"].shape)

        # print one sample for sanity check
        print("First sample token ids:", prepared["input_ids"][0][:20])

        # only inspect first 2 batches
        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()