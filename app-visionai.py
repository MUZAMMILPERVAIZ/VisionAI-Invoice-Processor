import concurrent.futures
from typing import Tuple
import os
import time
from pathlib import Path
import openai
import pandas as pd
import base64
import streamlit as st
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import vision
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import json
import io
import PyPDF2

openai.api_key = ""

def gptExtractor(ocr_text):
    system_role = '''
    Analyze the provided OCR text, identify key-value pairs corresponding to the entities below and return a dictionary in JSON format. If a field value cannot be found, replace it with 'None'. The following fields should be extracted:

    - "Supplier Name"
    - "Customer Name"
    - "TRN of Supplier"
    - "TRN of Customer"
    - "Address of Supplier"
    - "Address of Customer"
    - "Date of Invoice"
    - "Invoice Number"
    - "Payment due date"
    - "Description of goods / services"
    - "Net Amount"
    - "VAT Amount"
    - "Total Amount"
    - "Cross validated total amount"
    - "Email for correspondence"
    - "Tax Invoice"
    - "Tax"
    '''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": ocr_text},
        ]
    )
    return response['choices'][0]['message']['content']

def gptCorrector(ocr_text):
    system_role = '''
    Correct the provided dictionary by ensuring proper formatting, correct brackets, and braces. If a field is empty, null, or "", replace it with 'None'. 

    The following fields should be present:

    - "Supplier Name"
    - "Customer Name"
    - "TRN of Supplier"
    - "TRN of Customer"
    - "Address of Supplier"
    - "Address of Customer"
    - "Date of Invoice"
    - "Invoice Number"
    - "Payment due date"
    - "Description of goods / services"
    - "Net Amount"
    - "VAT Amount"
    - "Total Amount"
    - "Cross validated total amount"
    - "Email for correspondence"
    - "Tax Invoice"
    - "Tax"
    '''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": ocr_text},
        ]
    )
    return response['choices'][0]['message']['content']


def gptRefiner(gptText):
    system_role = '''
    From the provided dictionary, ensure that it only contains the following fields. If a field is missing, add it and set its value to 'None'. If a field's value is empty or "", replace it with 'None':

    - "Supplier Name"
    - "Customer Name"
    - "TRN of Supplier"
    - "TRN of Customer"
    - "Address of Supplier"
    - "Address of Customer"
    - "Date of Invoice"
    - "Invoice Number"
    - "Payment due date"
    - "Description of goods / services"
    - "Net Amount"
    - "VAT Amount"
    - "Total Amount"
    - "Cross validated total amount"
    - "Email for correspondence"
    - "Tax Invoice"
    - "Tax"
    '''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": gptText},
        ]
    )
    return response['choices'][0]['message']['content']




# VISION AI OCR

client = vision.ImageAnnotatorClient()


def detect_text(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_num = 0
    google_ocr_dict = {}

    for text in texts:
        # Create a sub-dictionary for each block of text
        google_ocr_dict[text_num] = {}
        # Get the coordinates of the bounding box around the text
        vertices = ([[vertex.x, vertex.y]
                     for vertex in text.bounding_poly.vertices])
        # Add the text and its coordinates to the dictionary
        google_ocr_dict[text_num]['text'] = text.description
        google_ocr_dict[text_num]['coords'] = vertices
        # Increment the text number
        text_num += 1

    with open("processed_image_new.json", "w") as json_file:
        json.dump(google_ocr_dict, json_file, indent=4)

    print(f"Created processed_image_new.json using Google OCR")
    return google_ocr_dict[0]["text"].replace("\n", " ")


def google_pdf_ocr(file_path, mime_type):
    document = file_path
    # mime_type = "application/pdf"
    mime_type = mime_type
    print('MIME TYPE: ', mime_type)
    with io.open(document, "rb") as f:
        content = f.read()
    input_config = {"mime_type": mime_type, "content": content}
    features = [{"type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
    pdf_reader=PyPDF2.PdfReader(document)
    num_pages=len(pdf_reader.pages)
    page_text={}
    num_pages = num_pages + 1
    pages = [[i for i in range(j, min(j + 5,num_pages), 1)] for j in range(1,num_pages,5)]
    ocr_text = {}
    for batch in pages:
        requests = [{"input_config": input_config, "features": features, "pages": batch}]
        response = client.batch_annotate_files(requests=requests)
        response = response.responses[0].responses
        page_no = 0
        for image_response in response:
            ocr_text1 = image_response.full_text_annotation.text

            ocr_text["Page_"+str(batch[page_no])] = ocr_text1
            page_no += 1

    return "\n".join(ocr_text.values())     # Join all pages' text with new line


import pandas as pd
import os

def csvWriter(refinedText, output_file):
    extracted_dict = eval(refinedText)
    df = pd.DataFrame([extracted_dict])

    if os.path.isfile(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    else:
        columns_order = ["Supplier Name", "Customer Name", "TRN of Supplier", "TRN of Customer",
                         "Address of Supplier", "Address of Customer", "Date of Invoice", "Invoice Number",
                         "Payment due date", "Description of goods / services ", "Net Amount ", "VAT Amount ",
                         "Total Amount ", "Cross validated total amount ", "Email for correspondence",
                         "Tax Invoice ", "Tax "]

        # Ensure all necessary columns exist in the DataFrame
        for col in columns_order:
            if col not in df.columns:
                df[col] = None

        # Reorder the columns
        df = df[columns_order]

    # Remove all columns in which all values are None or NaN
    df = df.dropna(how='all', axis=1)

    df.to_csv(output_file, index=False)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.csv">Click Here to Download</a>'
    return href



def pipeline(file_path, type, output_file):

    if type.startswith('image'):
        ocr_text = detect_text(file_path)
    else:
        ocr_text = google_pdf_ocr(
            file_path=file_path,
            mime_type=type,
        )
    gptExtractedText = gptExtractor(ocr_text)
    correctText = gptCorrector(gptExtractedText)
    refinedText = gptRefiner(correctText)
    csvWriter(refinedText, output_file)

    return refinedText


def process_file(uploaded_file) -> Tuple[str, str, str]:
    # Get file details
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}

    # Write the uploaded file to disk so it can be processed
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    output_file = f"_results.csv"

    # Run pipeline
    result = pipeline(file_path=uploaded_file.name, type=uploaded_file.type, output_file=output_file)

    # Delete the file after processing
    os.remove(uploaded_file.name)

    return result, uploaded_file.name, output_file


# Streamlit application
st.set_page_config(
    page_title="Document Processing App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",

)


def main():
    # Add a custom header with a different color
    st.markdown("""
                  <style>
                  .reportview-container {
                      flex-direction: row;
                      padding-top: 0rem;
                      padding-right: 0rem;
                      padding-left: 0rem;
                      padding-bottom: 0rem;
                  }
                  .sidebar .sidebar-content {
                      width: 50px;
                  }
                  .header {
                        color: white;
                        background-color: #000000;
                        padding: 10px;
                        margin-left: -80px;
                        text-align: center;
                        width: 100%;
                        position: fixed;
                        top: 40px;
                        z-index: 1000;
                }
                  .footer {
                      color: white;
                      background-color: #6883BC;
                      padding: 10px;
                      text-align: center;
                      margin-left: -80px;
                      width: 100%;
                      position: fixed;
                      bottom: 0px;
                      z-index: 1000;
                  }
                  .block-container {
                      padding-top: 150px; 
                      padding-bottom: 1000px; 
                      background-color: #6082B6;

                  }
                  .main{
                      background-color: #6082B6;
                  }
                
                    </style>
                  
                  </style>
                  """, unsafe_allow_html=True)

    st.markdown('<div class="header"><h2>Document Processing App</h2></div>', unsafe_allow_html=True)
    st.write("Enter your OpenAI key:")
    
    # Add an input field for the OpenAI key
    openai_key = st.text_input("OpenAI Key", "")
    
    # Update the OpenAI key in the openai package
    openai.api_key = openai_key
    
    # Add a file upload widget for Google credentials
    google_credentials = st.file_uploader("Upload Google Credentials JSON", type="json")
    
    # Check if Google credentials are uploaded and set the environment variable
    if google_credentials is not None:
        with open("credentials.json", "wb") as f:
            f.write(google_credentials.getbuffer())
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

    # User can upload multiple files
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

    # Process button
    if st.button('Process'):
        with st.spinner("Performing OCR..."):

            # Ensure files have been uploaded
            if not uploaded_files:
                st.warning('Please upload a file.')
                return

            # # Progress bar
            # progress_bar = st.progress(0)

            # Output area
            st.subheader("Extracted Data:")

            # Start the timer
            start_time = time.time()

            # Create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Iterate through uploaded files
                results = list(executor.map(process_file, uploaded_files))

                # for i, (result, uploaded_file_name, output_file) in enumerate(results):
                    # Display results
                    # st.write(result)

                    # Update the progress bar
                    # progress_bar.progress((i + 1) / len(uploaded_files))

            # Stop the timer
            end_time = time.time()

            # Calculate and display the total time taken
            total_time = end_time - start_time
            st.write(f'Total time taken: {total_time:.2f} seconds')

            # Completion message
            st.success('Document Scanned Successfully!')

            df = pd.read_csv('_results.csv')
            st.dataframe(df)  # Use st.dataframe for full width
    if st.button('Download'):
        href = get_binary_file_downloader_html('_results.csv', 'Processed Results')
        st.markdown(href, unsafe_allow_html=True)



    # Add a custom footer with a different color
    # st.markdown('<div class="footer"><h4>© 2023 Document Processing App. All rights reserved.</h4></div>',
    #             unsafe_allow_html=True)


if __name__ == '__main__':
    main()
