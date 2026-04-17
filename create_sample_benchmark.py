"""
Create sample benchmark data for evaluation
Generates synthetic benchmark data based on the corpus
"""

import json
from pathlib import Path


def create_sample_benchmark():
    """Create sample benchmark data in data/benchmark/"""
    
    benchmark_dir = Path("data/benchmark")
    benchmark_dir.mkdir(exist_ok=True)
    
    # Sample queries based on the EEG paper
    sample_queries = [
        "What is the main topic of this paper?",
        "What methods are used for subject identification?",
        "What datasets are used in the experiments?",
        "What are the results of the subject identification accuracy?",
        "How are EEG signals processed for identification?",
        "What is imagined speech?",
        "What are the advantages of using EEG for identification?",
        "What are the preprocessing steps mentioned?",
        "How many subjects participated in the study?",
        "What is the confidence level of the results?",
    ]
    
    # Sample reference answers
    sample_answers = [
        "The paper investigates the potential of using EEG (Electroencephalogram) signals for subject identification during imagined speech, exploring methods that don't require training or feedback.",
        "Methods include preprocessing of EEG data, feature extraction, and subject identification using signals during imagined speech, as well as verification protocols.",
        "The paper uses EEG datasets from experiments with subjects performing imagined speech tasks. Datasets include VEP (Visual Evoked Potential) datasets and other EEG recordings.",
        "The results show rank-1 identification accuracies with corresponding 95% confidence intervals varying by dataset and methodology used.",
        "EEG signals are preprocessed through various stages including filtering, artifact removal, and normalization before feature extraction and identification.",
        "Imagined speech refers to the mental activity of thinking about speaking without actually vocalizing, which produces distinctive EEG signal patterns.",
        "EEG-based identification offers non-invasive subject identification, can work without training feedback, and provides a secure method for identification and authentication.",
        "The preprocessing steps include signal filtering, artifact removal, normalization, and feature extraction from the raw EEG recordings.",
        "The study includes multiple subjects, with separate sessions (session 1 and session 2) for each subject to test consistency and robustness.",
        "The confidence intervals provided show the statistical reliability of the identification accuracies achieved by the methods, ranging across different experimental conditions.",
    ]
    
    # Sample corpus - simplified versions of document chunks
    sample_corpus = {
        "doc_0": "Subject Identification from Electroencephalogram (EEG) Signals During Imagined Speech. The paper investigates the potential of using electrical brain signals for secure identification and authentication of individuals.",
        "doc_1": "Abstract: We investigate the potential of using EEG signals during imagined speech for subject identification. Methods that do not require training or feedback are explored.",
        "doc_2": "Section I provides an introduction to EEG-based biometrics and subject identification. EEG signals offer unique patterns for each individual.",
        "doc_3": "Section II describes the EEG datasets used in this work, including VEP datasets and recordings from subjects performing imagined speech tasks.",
        "doc_4": "Section III details the preprocessing approaches used for EEG signal processing, including filtering, artifact removal, and normalization.",
        "doc_5": "Imagined speech refers to thinking about speaking without actual vocalization. This mental activity produces distinctive EEG signal patterns.",
        "doc_6": "The paper implements methods for practical subject identification without requiring training or feedback from subjects.",
        "doc_7": "Table results show average rank-1 identification accuracies and corresponding 95% confidence intervals for the VEP EEG dataset.",
        "doc_8": "The study includes multiple subjects, with separate sessions to assess robustness and consistency of the identification methods.",
        "doc_9": "Discussion of advantages: EEG-based identification is non-invasive, does not require prior training, and can create secure authentication systems.",
    }
    
    # Save queries
    queries_file = benchmark_dir / "queries.json"
    with open(queries_file, 'w') as f:
        json.dump(sample_queries, f, indent=2)
    print(f"✓ Created {queries_file} with {len(sample_queries)} queries")
    
    # Save reference answers
    answers_file = benchmark_dir / "ansers.json"  # Keep the typo to match expected filename
    with open(answers_file, 'w') as f:
        json.dump(sample_answers, f, indent=2)
    print(f"✓ Created {answers_file} with {len(sample_answers)} reference answers")
    
    # Save corpus
    corpus_file = benchmark_dir / "corpus.json"
    with open(corpus_file, 'w') as f:
        json.dump(sample_corpus, f, indent=2)
    print(f"✓ Created {corpus_file} with {len(sample_corpus)} documents")
    
    print("\n✓ Sample benchmark data created successfully!")
    print(f"\nBenchmark data saved to: {benchmark_dir}")
    print("\nYou can now run the evaluation with:")
    print("  python3 evaluate.py")


if __name__ == "__main__":
    create_sample_benchmark()
