import os
from document_reader import DocumentReader
from publication_chunker import PublicationChunker
from vector_db import VectorDB 
from rag_qa import RAGQA
from publication_translator import PublicationTranslator
from publication_summarizer import PublicationSummarizer

def main():
    """
    Main function to demonstrate document processing functionality.
    """
    # Initialize the document reader
    reader = DocumentReader()
    
    # Directory containing Dr. X's publications
    publications_dir = "publications"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(publications_dir):
        os.makedirs(publications_dir)
        print(f"Created directory: {publications_dir}")
        print("Please place Dr. X's publications in this directory.")
        return
    
    # Get all files in the publications directory
    files = [f for f in os.listdir(publications_dir) if os.path.isfile(os.path.join(publications_dir, f))]
    
    if not files:
        print(f"No files found in {publications_dir} directory.")
        print("Please add some files to process.")
        return
    
    print(f"Found {len(files)} files in {publications_dir} directory.")
    
    # Initialize VectorDB
    vector_db = VectorDB()
    
    # Mode selection
    print("\n=== Select Mode ===")
    print("1. Question & Answer")
    print("2. Document Translation")
    print("3. Document Summarization")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == '1':
        # Process files and add to vector database only for Q&A mode
        for file_name in files:
            file_path = os.path.join(publications_dir, file_name)
            extension = file_name.split('.')[-1].lower()

            if extension in reader.supported_extensions:
                print(f"\nProcessing: {file_name}")
                try:
                    chunker = PublicationChunker()
                    chunks = chunker.chunk_publication(file_path)
                    
                    # Add chunks to vector database
                    vector_db.add_to_db(chunks)
                    print(f"Added {len(chunks)} chunks to vector database")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
            else:
                print(f"Skipping unsupported file: {file_name}")
        
        print(f"\nTotal chunks in vector database: {vector_db.get_db_size()}")

        # Initialize RAG QA system
        rag = RAGQA(vector_db)

        # Interactive Q&A session
        print("\n=== Interactive Q&A Session ===")
        print("Type 'exit' to end the session.")
        
        while True:
            question = input("\nYour question: ")
            
            if question.lower() == 'exit':
                print("Exiting Q&A session. Goodbye!")
                break
            
            # Process the question and get answer
            answer = rag.answer_question(question)
            print("\nAnswer:", answer)
            print("\n" + "-"*50)
    
    elif choice == '2':
        # Initialize translator
        translator = PublicationTranslator()
        
        # Select file to translate
        print("\nAvailable files:")
        for i, file_name in enumerate(files):
            print(f"{i+1}. {file_name}")
        
        file_choice = input("\nSelect file to translate (enter number): ").strip()
        try:
            selected_index = int(file_choice) - 1
            if 0 <= selected_index < len(files):
                selected_file = files[selected_index]
                file_path = os.path.join(publications_dir, selected_file)
                
                # Select target language
                print("\nTarget languages:")
                print("1. English")
                print("2. Arabic")
                lang_choice = input("Select target language (1 or 2): ").strip()
                target_lang = "english" if lang_choice == '1' else "arabic"
                
                # Translate and save
                translated = translator.translate_document(file_path, target_lang)
                output_path = os.path.join("translations", f"translated_{selected_file}")
                saved_path = translator.save_translated_document(translated, output_path)
                print(f"\nTranslation saved to: {saved_path}")
            else:
                print("Invalid file selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    elif choice == '3':
        # Initialize summarizer
        summarizer = PublicationSummarizer()
        
        # Select file to summarize
        print("\nAvailable files:")
        for i, file_name in enumerate(files):
            print(f"{i+1}. {file_name}")
        
        file_choice = input("\nSelect file to summarize (enter number): ").strip()
        try:
            selected_index = int(file_choice) - 1
            if 0 <= selected_index < len(files):
                selected_file = files[selected_index]
                file_path = os.path.join(publications_dir, selected_file)
                
                # Select summarization technique
                print("\nSummarization techniques:")
                print("1. Extractive")
                print("2. Abstractive")
                print("3. Hybrid")
                technique_choice = input("Select technique (1, 2, or 3): ").strip()
                
                technique_map = {
                    "1": "extractive",
                    "2": "abstractive",
                    "3": "hybrid"
                }
                
                if technique_choice in technique_map:
                    technique = technique_map[technique_choice]
                    print(f"\nGenerating {technique} summary...")
                    
                    # Generate summary
                    summary_result = summarizer.summarize_document(file_path, technique)
                    
                    # Create summaries directory if it doesn't exist
                    summaries_dir = "summaries"
                    if not os.path.exists(summaries_dir):
                        os.makedirs(summaries_dir)
                    
                    # Save summary
                    output_path = os.path.join(summaries_dir, f"{technique}_summary_{selected_file}")
                    saved_path = summarizer.save_summary(summary_result["summary"], output_path)

                    print(f"\nSummary saved to: {saved_path}")
                    
                    # Ask if user wants to evaluate the summary
                    eval_choice = input("\nDo you want to evaluate this summary against a reference? (y/n): ").strip().lower()
                    if eval_choice == 'y':
                        ref_path = input("Enter path to reference summary file: ").strip()
                        try:
                            with open(ref_path, 'r', encoding='utf-8') as f:
                                reference_summary = f.read()
                            
                            # Evaluate summary
                            rouge_scores = summarizer.evaluate_summary(reference_summary, summary_result["summary"])
                            
                            # Display evaluation results
                            print("\nROUGE Evaluation Results:")
                            for metric, scores in rouge_scores.items():
                                if isinstance(scores, dict) and "f" in scores:
                                    print(f"  {metric}: {scores['f']:.4f}")
                        except Exception as e:
                            print(f"Error evaluating summary: {str(e)}")
                else:
                    print("Invalid technique selection.")
            else:
                print("Invalid file selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()