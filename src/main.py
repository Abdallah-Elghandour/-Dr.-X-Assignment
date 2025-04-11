import os
from document_reader import DocumentReader

def main():
    """
    Main function to demonstrate the document reader functionality.
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
    
    # Process each file
    for file_name in files:
        file_path = os.path.join(publications_dir, file_name)
        extension = file_name.split('.')[-1].lower()
        
        if extension in reader.supported_extensions:
            print(f"\nProcessing: {file_name}")
            try:
                result = reader.read_document(file_path)
                
                # Print some information about the extracted content
                print(f"Source: {result['source']}")
                print(f"Extension: {result['extension']}")
                print(f"Total pages: {len(result['pages'])}")
                
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        else:
            print(f"Skipping unsupported file: {file_name}")

if __name__ == "__main__":
    main()