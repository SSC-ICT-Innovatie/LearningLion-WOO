import logging
import os
import settings
import streamlit as st
from datetime import date
from loguru import logger
from PIL import Image

import utils as ut
from ingest.ingester import Ingester
from query.querier import Querier
from summarize.summarizer import Summarizer


def click_go_button():
    """
    Sets session state of GO button clicked to True
    """
    st.session_state['is_GO_clicked'] = True


@st.cache_data
def create_and_show_summary(my_summary_type,
                            my_folder_path_selected,
                            my_folder_name_selected,
                            my_vectordb_folder_path_selected):
    summarization_method = "Map_Reduce" if my_summary_type == "Short" else "Refine"
    # for each file in content folder
    with st.expander(f"{my_summary_type} summary"):
        for file in os.listdir(my_folder_path_selected):
            if os.path.isfile(os.path.join(my_folder_path_selected, file)):
                summary_name = os.path.join(my_folder_path_selected,
                                            "summaries",
                                            file + "_" + str.lower(my_summary_type) + ".txt")
                # if summary does not exist yet, create it
                if not os.path.isfile(summary_name):
                    my_spinner_message = f'''Creating summary for {file}.
                                        Depending on the size of the file, this may take a while. Please wait...'''
                    with st.spinner(my_spinner_message):
                        summarizer = Summarizer(content_folder=my_folder_path_selected,
                                                collection_name=my_folder_name_selected,
                                                summary_method=summarization_method,
                                                vectordb_folder=my_vectordb_folder_path_selected)
                        summarizer.summarize()
                # show summary
                st.write(f"**{file}:**\n")
                with open(file=summary_name, mode="r", encoding="utf8") as f:
                    st.write(f.read())
                    st.divider()


def display_chat_history():
    """
    Shows the complete chat history
    """
    length = len(st.session_state['messages'])
    for index, message in enumerate(st.session_state['messages']):
        if message["role"] == "results":
            # Only expand it on the most recent entry
            show_results(message["content"], expand=index==length-1)
            continue
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            continue
    logger.info("Executed display_chat_history()")
    
def show_results(documents, expand=True):
    with st.expander("Results", expanded=expand):
        for index, tuple in enumerate(documents):
            document, score = tuple
            retrieval_method_info = f' , retrieval method: {document.metadata.get("retrieval_method")}' if document.metadata.get("retrieval_method") else ""
            if index != 0:
                st.markdown("---")
            document_link = f', document: [link]({document.metadata["documents_dc_source"]})' if document.metadata.get("documents_dc_source") else ""
            dossier_link = f', source: [link](https://pid.wooverheid.nl/?pid={document.metadata["foi_dossierId"]})' if document.metadata.get("foi_dossierId") else ""
            st.markdown(f'''**Document id: {document.metadata['foi_documentId']},
                        page: {document.metadata['page_number']},
                        chunk: {document.metadata['chunk']},
                        dossier: {document.metadata['dossiers_dc_title']},
                        score: {score:.4f}{retrieval_method_info}{document_link}{dossier_link}**''')
            st.markdown(f"{document.page_content}")

@st.cache_data
def folderlist_creator():
    """
    Creates a list of folder names (without path).
    Folder names are found in DOC_DIR (see settings).
    """
    folders = []
    for folder_name in os.listdir(settings.DOC_DIR):
        folder_path = os.path.join(settings.DOC_DIR, folder_name)
        if os.path.isdir(folder_path):
            folders.append(folder_name)
    logger.info("Executed folderlist_creator()")
    return folders


def folder_selector(folders):
    # Select source folder with docs
    my_folder_name_selected = st.sidebar.selectbox("label=folder_selector", options=folders, label_visibility="hidden")
    logger.info(f"folder_name_selected is now {my_folder_name_selected}")
    # get associated source folder path and vectordb path
    my_folder_path_selected, my_vectordb_folder_path_selected = ut.create_vectordb_name(my_folder_name_selected)
    logger.info(f"vectordb_folder_path_selected is now {my_vectordb_folder_path_selected}")
    if my_folder_name_selected != st.session_state['folder_selected']:
        st.session_state['is_GO_clicked'] = False
    # set session state of selected folder to new source folder
    st.session_state['folder_selected'] = my_folder_name_selected
    return my_folder_name_selected, my_folder_path_selected, my_vectordb_folder_path_selected

@st.cache_resource
def check_vectordb(_my_querier, my_folder_name_selected, my_folder_path_selected, my_vectordb_folder_path_selected):
    # If a folder is chosen that is not equal to the last known source folder
    if folder_name_selected != st.session_state['folder_selected']:
        # set session state of is_GO_clicked to False (will be set to True when OK button is clicked)
        st.session_state['is_GO_clicked'] = False
        # clear all chat messages on screen and in Querier object
        st.session_state['messages'] = []
        _my_querier.clear_history()
    # When the associated vector database of the chosen content folder doesn't exist with the settings as given
    # in settings.py, create it first
    if not os.path.exists(my_vectordb_folder_path_selected):
        logger.info("Creating vectordb")
        my_spinner_message = f'''Creating vector database for folder {my_folder_name_selected}.
        Depending on the size, this may take a while. Please wait...'''
    else:
        logger.info("Updating vectordb")
        my_spinner_message = f'''Checking if vector database needs an update for folder {my_folder_name_selected}.
        This may take a while, please wait...'''
    with st.spinner(my_spinner_message):
        ingester = Ingester(my_folder_name_selected,
                            my_folder_path_selected,
                            my_vectordb_folder_path_selected)
        ingester.ingest()

    if not _my_querier.chain:
        # create a new chain based on the new source folder
        _my_querier.make_chain(my_folder_name_selected, my_vectordb_folder_path_selected)
        # set session state of selected folder to new source folder
        st.session_state['folder_selected'] = my_folder_name_selected
        logger.info("Executed check_vectordb")
        
    
def update_filters(my_querier, selected_publisher, start_date, end_date):
    filters = []
    if selected_publisher and selected_publisher != "--None--":
        filters.append({"dossiers_dc_publisher_name": selected_publisher})

    # Initialize date filters as a separate list to hold all date-related conditions
    date_filters = []
    if start_date:
        date_filters.append({"$or": [
            {"dossiers_year": {"$gt": start_date.year}},
            {"$and": [{"dossiers_year": start_date.year}, {"dossiers_month": {"$gt": start_date.month}}]},
            {"$and": [{"dossiers_year": start_date.year}, {"dossiers_month": start_date.month}, {"dossiers_day": {"$gt": start_date.day}}]}
        ]})

    if end_date:
        date_filters.append({"$or": [
            {"dossiers_year": {"$lt": end_date.year}},
            {"$and": [{"dossiers_year": end_date.year}, {"dossiers_month": {"$lt": end_date.month}}]},
            {"$and": [{"dossiers_year": end_date.year}, {"dossiers_month": end_date.month}, {"dossiers_day": {"$lte": end_date.day}}]}
        ]})

    # Add the combined date filters to the main filters list, if any date conditions exist
    if date_filters:
        # If there are both start_date and end_date filters, they need to be combined logically
        if len(date_filters) > 1:
            filters.append({"$and": date_filters})
        else:
            filters.extend(date_filters)

    # If there are multiple filters, wrap them in $and, otherwise, just use the single filter directly
    final_filters = {"$and": filters} if len(filters) > 1 else filters[0] if filters else {}

    my_querier.filters = final_filters
    st.session_state['filters_saved'] = True
    st.success("Filters are successfully saved!")
    log_action("Filters applied", publisher=selected_publisher, start_date=start_date, end_date=end_date)
    logger.info("Filters updated: ", my_querier.filters)

def simplify_response(response):
    simplified_docs = []
    for document, score in response['source_documents']:
        simplified_doc = {
            'chunk': document.metadata['chunk'],
            'documents_dc_source': document.metadata.get('documents_dc_source', 'N/A'),
            'foi_documentId': document.metadata.get('foi_documentId', 'N/A'),
            'foi_dossierId': document.metadata.get('foi_dossierId', 'N/A'),
            'dossiers_dc_title': document.metadata.get('dossiers_dc_title', 'N/A'),
            'retrieval_method': document.metadata.get('retrieval_method', 'N/A'),
            'score': score
        }
        simplified_docs.append(simplified_doc)
    return simplified_docs

def handle_query(my_querier, my_prompt: str):
    log_action("Query", prompt=my_prompt)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(my_prompt)
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": my_prompt})
    with st.spinner("Loading..."):
        # Generate a response
        response = my_querier.ask_question(my_prompt)
        compact_response = simplify_response(response)
        log_action("Response", response=compact_response)
    if len(response["source_documents"]) > 0:
        st.session_state['messages'].append({"role": "results", "content": response["source_documents"]})
    else:
        logger.warning("No source documents found relating to the question")
    logger.info("Executed handle_query(querier, prompt)")
    st.rerun()

@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    Also prepares the sidebar with folder list
    """
    imagecol, headercol = st.columns([0.3, 0.7])
    logo_image = Image.open(settings.APP_LOGO)
    with imagecol:
        st.image(logo_image, width=250)
    with headercol:
        st.header(settings.APP_HEADER)
    # Custom CSS to have white expander background
    st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: white;
            color: black; # Adjust this for expander header color
        }
        .streamlit-expanderContent {
            background-color: white;
            color: black; # Expander content color
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
    # with st.sidebar.expander("User manual"):
    #     # read app explanation from file explanation.txt
    #     with open(file=settings.APP_INFO, mode="r", encoding="utf8") as f:
    #         explanation = f.read()
    #     st.markdown(body=explanation, unsafe_allow_html=True)
    #     st.image("./images/multilingual.png")
    # st.sidebar.divider()
    # Sidebar text for folder selection
    st.sidebar.title("Select your WOO folder")
    logger.info("Executed initialize_page()")
    

def initialize_session_state():
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False
    if 'folder_selected' not in st.session_state:
        st.session_state['folder_selected'] = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'filters_saved' not in st.session_state:
        st.session_state['filters_saved'] = True

@st.cache_resource
def initialize_querier():
    """
    Create a Querier object
    """
    my_querier = Querier()
    logger.info("Executed initialize_querier()")
    return my_querier

@st.cache_resource
def initialize_logger():
    if settings.LOGGING:
        # Create logging folder if it doesn't exist
        if not os.path.exists(settings.LOGGING_FOLDER):
            os.makedirs(settings.LOGGING_FOLDER)
        current_time = ut.get_timestamp().replace(" ", "_").replace(":", "-")
        log_filename = f"{settings.LOGGING_FOLDER}/log_{current_time}.log"
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.FileHandler(log_filename, 'a', 'utf-8')])
        
def log_action(action, **kwargs):
    if settings.LOGGING:
        # Prepare additional details as a string
        details = ', '.join(f"{key}={value}" for key, value in kwargs.items())
        # Log action along with additional details
        logging.info(f"{action} - {details}" if details else action)

def set_page_config():
    favicon = Image.open("images/favicon.ico")
    st.set_page_config(page_title="SSC-ICT WOO-RAG",
                       page_icon=favicon,
                       layout='wide',
                       initial_sidebar_state='auto')
    logger.info("Executed set_page_config()")

### MAIN PROGRAM ####
# Set page configuration and initialize all necessary settings
set_page_config()
initialize_session_state()
initialize_page()
initialize_logger()
# Creation of Querier object, only once per session
querier = initialize_querier()
# Chosen folder and associated vector database
folder_name_selected, folder_path_selected, vectordb_folder_path_selected = folder_selector(folderlist_creator())

# Create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True
st.sidebar.button("GO", type="primary", on_click=click_go_button)

# Set clear warning for the user when logging is enabled
if settings.LOGGING and not st.session_state['is_GO_clicked']:
    st.warning("Warning: Logging is enabled for this session. All actions are logged.")

# Start a conversation when a folder is selected and selection is confirmed with "GO" button
if st.session_state['is_GO_clicked']:
    # Create or update vector database if necessary, only once per session
    check_vectordb(querier, folder_name_selected, folder_path_selected, vectordb_folder_path_selected)
    st.sidebar.divider()
    with st.sidebar.expander("Search filters"):
        def on_change():
            st.session_state['filters_saved'] = False

        publishers = ["--None--"] + querier.get_woo_publisher()
        selected_publisher = st.selectbox("Filter on Publisher", publishers, on_change=on_change)
    
        # Date input for date range search
        start_date = st.date_input("After Date", value=None, min_value=date(2010, 1, 1), max_value=date.today(), key='start_date', on_change=on_change)
        end_date = st.date_input("Before Date", value=None, min_value=date(2010, 1, 1), max_value=date.today(), key='end_date', on_change=on_change)
        
        # Assuming you have a button to apply filters
        apply_filters = st.button("Apply Filters", key="apply_filters")
        if apply_filters:
            update_filters(querier, selected_publisher, start_date, end_date)
            
        if not st.session_state['filters_saved']:
            st.error("You have unsaved filters!")
            
    # summary_type = st.sidebar.radio(
    #     "Start with summary?",
    #     ["No", "Short", "Long"],
    #     captions=["No, start the conversation", "Quick but lower quality", "Slow but higher quality"],
    #     index=0)
    # # if a short or long summary is chosen
    # if summary_type in ["Short", "Long"]:
    #     # show the summary at the top of the screen
    #     create_and_show_summary(summary_type, folder_path_selected, folder_name_selected, vectordb_folder_path_selected)
            
    # Display chat messages from history
    display_chat_history()
    
    # React to user input if a question has been asked
    if prompt := st.chat_input("Your question", key="chat_input"):
        handle_query(querier, prompt)

    # If button "Clear History" is clicked
    if len(st.session_state['messages']) > 0:
        # Show button "Clear History"
        clear_messages_button = st.button("Clear History", key="clear")
        if clear_messages_button:
            # Clear all chat messages on screen and in Querier object
            # NB: session state of "is_GO_clicked" and "folder_selected" remain unchanged
            st.session_state['messages'] = []
            querier.clear_history()
            logger.info("Clear History button clicked")
    # If no messages yet, give suggestions
    else:
        st.markdown("##")
        st.subheader("Voorbeeld prompt:")
        example_question_1 = "Wat is de hoogste politiesalaris in Nederland?"
        example_question_2 = "Wat wordt er gedaan om meer energiezuinig om te gaan in Nederland?"
        example_question_3 = "Welke beleidsdocumenten zijn er met betrekking tot het milieuwetgevingsbeleid in Den Haag?"
        if st.button(example_question_1, key="example_question_1"):
            handle_query(querier, example_question_1)
        if st.button(example_question_2, key="example_question_2"):
            handle_query(querier, example_question_2)
        if st.button(example_question_3, key="example_question_3"):
            handle_query(querier, example_question_3)
            