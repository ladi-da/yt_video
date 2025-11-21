# -*- coding: utf-8 -*-
# <<< START: IMPORT STATEMENTS AND HELPER FUNCTIONS >>> 
import streamlit as st
import logging
import requests
import pandas as pd
import json
import time
import yt_dlp # Import the yt-dlp library
import os
import random
import string
import datetime
import urllib.parse
import textwrap
import tempfile
from io import BytesIO
import anthropic # Make sure anthropic client is installed: pip install anthropic
import moviepy.audio.fx.all as afx
from google import genai
import itertools
# Ensure MoviePy is installed: pip install moviepy
# Ensure Pillow is installed: pip install Pillow
# Ensure pydub is installed: pip install pydub
# Ensure numpy is installed: pip install numpy
import numpy as np
import cv2
import pytesseract
from google.oauth2 import service_account
from googleapiclient.discovery import build
from zoneinfo import ZoneInfo

# --- Try importing moviepy components with error handling ---
# Cache for resolved yt-dlp direct URLs to avoid refetching for the same video ID within a session
if 'resolved_vid_urls' not in st.session_state:
  st.session_state['resolved_vid_urls'] = {} # youtube_url: dlp_info_dict

logging.basicConfig(level=logging.INFO, force=True)
 

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips, TextClip,CompositeAudioClip
    )
    import moviepy.video.fx.all as vfx
    # This specific import might be less common or part of older versions, handle potential error
    try:
        import moviepy.video.fx.resize as moviepy_resize
    except ImportError:
        logging.info("Note: moviepy.video.fx.resize not found (may be integrated in newer MoviePy).")
        moviepy_resize = None # Define as None if not found

    from pydub import AudioSegment
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    st.error(f"Missing required library: {e}. Please install MoviePy, Pillow, pydub, and numpy.")
    st.stop() # Stop the app if essential libraries are missing

# --- API Clients & Services ---
from openai import OpenAI
# Ensure boto3 is installed: pip install boto3
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:
    st.error("Missing required library: boto3. Please install it.")
    st.stop()
def create_combos(items):
    all_combos = []
    for r in range(1, len(items)+1):
        all_combos.extend(itertools.combinations(items, r))
    return all_combos


# --- Configuration ---
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3/search"
MAX_RESULTS_PER_QUERY = 100 # How many results to fetch *per term* from YouTube API
YT_DLP_FETCH_TIMEOUT = 30 # Increased timeout for potentially slower connections
DEFAULT_TTS_VOICE = "sage" # Default voice for TTS
GOOGLE_TIKTOK_CX_ID = st.secrets.get("GOOGLE_TIKTOK_CX_ID", "331dbbc80d31342af")
GOOGLE_PINTEREST_CX_ID = st.secrets.get("GOOGLE_PINTEREST_CX_ID", GOOGLE_TIKTOK_CX_ID)

AUTO_TERMS_PROMPT = """You are a Viral Video Ad Scout tasked with finding top YouTube Shorts search terms that reveal
                                                 visually compelling, user-generated content perfect for remixing into high-performing Facebook video ads.
                                                 Given a topic, return 4 concise,  remix-ready search terms (one being the topic itself),
                                                 each ending with #shorts and separated by | """
AUTO_TERMS_PROMPT = """You are a Viral Video Ad Scout creating high-performing Facebook Reels-style video ads with TTS overlay.
Your job is to source YouTube Shorts search terms that lead to visually compelling, user-generated content for remixing into short click-driven videos.

Given a topic, return 4 concise, remix-ready YouTube Shorts search terms, each ending in #shorts, and separated by |.

Requirements:
- One term must reflect the original topic, reworded for visual appeal and curiosity.
- Focus on search terms that lead to aspirational or aesthetic content — show the 'pretty side' of the topic:
  • Beautiful visuals, luxury moments, new or clean products, satisfying transformations.
  • Think: “new catamaran tour,” “luxury tiny home,” “dream RV inside” — not “repossessed boat for sale.”
- All terms must be:
  • 2–4 words long.
  • Remixable into a TTS Facebook video (i.e. clearly visual, engaging, scroll-stopping).
  • Based on UGC-style videos or Shorts content people naturally upload.
  • Emotionally resonant: provoke curiosity, awe, joy, or satisfaction.

Examples of strong terms:
- New catamaran tour
- Dream boat interior
- Luxury yacht walk
- Satisfying boat renovation

Avoid:
- SEO-style or commercial phrases like “for sale,” “buy now,” or “discount.”
- Overly specific or boring search terms like “repossessed catamarans for sale.”
- Any term longer than 4 words.

return just the requested string, no intros no explanations
"""


# --- Font Path Handling ---
# Use an absolute path or ensure the font is in the same directory as the script
# For Streamlit Cloud, you might need to include the font in your repository
# and reference it relative to the script's location.
# If running locally, ensure the path is correct.
MONTSSERAT_FONT_FILENAME = "Montserrat-Bold.ttf"

if 'sheet_appended' not in st.session_state:
    st.session_state.sheet_appended = False

# Try to find the font in the current directory or a subdirectory
if os.path.exists(MONTSSERAT_FONT_FILENAME):
    SUBTITLE_FONT_PATH = MONTSSERAT_FONT_FILENAME
elif __file__ and os.path.exists(os.path.join(os.path.dirname(__file__), MONTSSERAT_FONT_FILENAME)):
     SUBTITLE_FONT_PATH = os.path.join(os.path.dirname(__file__), MONTSSERAT_FONT_FILENAME)
else:
    st.warning(f"Font '{MONTSSERAT_FONT_FILENAME}' not found. Subtitles might use default font.", icon="⚠️")
    SUBTITLE_FONT_PATH = None # Will use default font later if None

SUBTITLE_FONT_SIZE = 50 # Adjust as needed
SUBTITLE_WORDS_PER_GROUP = 2 # Group words for subtitles
SUBTITLE_COLOR = '#FFFF00' # Yellow
SUBTITLE_BG_COLOR = 'rgba(0, 0, 0, 0.6)' # Semi-transparent black
st.set_page_config(layout="wide", page_title="YouTube Video Generator", page_icon="🎥")
SCRIPT_VER_OPTIONS =create_combos(["default", "default_v2", "1st_person","default_clean" ])
BG_VER_OPTIONS =[ False,"sunrise","funk","ukulele", "mix"]
TTS_VOICE_OPTIONS = create_combos(['sage','redneck','announcer','sage uk','announcer uk'])
# Consistent Backblaze path for generated creatives
S3_VIDSLIDE_PREFIX = "creative-cloud/yt-video"
# --- Load Secrets ---
s3_endpoint_url = st.secrets.get("S3_ENDPOINT_URL")
try:
    GEMINI_API_KEY =st.secrets.get("GEMINI_API_KEY")

    youtube_api_key_secret = st.secrets["YOUTUBE_API_KEY"]
    openai_api_key = st.secrets["GPT_API_KEY1"]
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"] # Added Anthropic Key
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    s3_bucket_name = st.secrets["S3_BUCKET_NAME"]
    s3_region = st.secrets["AWS_REGION"]
    COOKIE_FILE_PATH = st.secrets.get("YOUTUBE_COOKIE_PATH") # Optional cookie path
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please configure secrets.", icon="🚨")
    st.stop()

# --- Initialize Clients ---
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=openai_api_key)

openai_client = get_openai_client()

# Note: Anthropic client doesn't benefit as much from @st.cache_resource
# as it's typically lightweight to initialize. Initialize it where needed or globally.
# client = anthropic.Anthropic(api_key=anthropic_api_key) # Or initialize inside claude function

@st.cache_resource
def get_s3_client():
    try:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=s3_region,
            endpoint_url=s3_endpoint_url.strip() if s3_endpoint_url else None
        )
    except NoCredentialsError:
        st.error("AWS Credentials not found or invalid in secrets.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"Error initializing S3 client: {e}", icon="🚨")
        return None

s3_client = get_s3_client()

# --- Patched Resizer (Optional) ---
try:
    if moviepy_resize: # Check if the module was imported
        def patched_resizer(pilim, newsize):
            if isinstance(newsize, (list, tuple)):
                newsize = tuple(int(dim) for dim in newsize)
            elif isinstance(newsize, (int, float)):
                if hasattr(pilim, "shape"): # numpy array
                    orig_height, orig_width = pilim.shape[:2]
                else: # PIL image
                    orig_width, orig_height = pilim.size
                newsize = (int(orig_width * newsize), int(orig_height * newsize))

            if not isinstance(pilim, Image.Image):
                pilim = Image.fromarray(pilim)

            # Use LANCZOS for resizing - generally good quality
            resized = pilim.resize(newsize, Image.Resampling.LANCZOS)
            return np.array(resized)

        moviepy_resize.resizer = patched_resizer
        logging.info("Applied patched resizer.")
    else:
         logging.info("Skipping resizer patch: moviepy.video.fx.resize not found.")
except Exception as e:
    logging.info(f"Could not apply patched resizer: {e}")
    pass # Continue without patch

# --- Helper Function: create_topic_summary_dataframe ---

def google_sheets_append_df(spreadsheet_id,range_name, df_data_input ):
    # Load credentials from Streamlit secrets
    credentials_dict = st.secrets["gcp_service_account"]
    # st.text(credentials_dict)
    creds = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )


    service = build("sheets", "v4", credentials=creds)
    time_now = [datetime.datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d %H:%M")]
    df_list = df_data_input.values.tolist()
    for idx in range(len(df_list)):
        df_list[idx] = time_now + df_list[idx]
        
    body = {"values": df_list}
    try:
        # Append the row
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body=body
        ).execute()

        st.success(f"Row added: {result['updates']['updatedRange']}")
    except Exception as e:
        st.text(f"error google_sheets_append_row:  {e}")
def blur_subtitles_in_video_unified(
    video_path,
    output_path, # This MUST be different from video_path
    sample_time_sec=5.0,
    ocr_min_confidence=40,
    ocr_y_start_ratio=0.70,
    ocr_padding=15,
    blur_kernel_size=(51, 51),
    tesseract_cmd_path=None,
    debug_save_frames=False
):
    """
    Loads a video, determines a subtitle bounding box from a sample frame using
    Pytesseract, and then blurs this region throughout the video.
    Saves the processed video to output_path.
    """

    # --- Nested Helper Function: Determine Subtitle Bounding Box ---
    def _determine_subtitle_bbox_from_frame(
        frame_pil_local,
        min_confidence_local,
        expected_y_start_ratio_local,
        padding_local
    ):
        try:
            # CRITICAL: Removed Streamlit st.image and st.text calls
            # For debugging OCR:
            if debug_save_frames: # You can control this with the main function's arg
                try:
                    frame_pil_local.save("debug_ocr_sample_input_for_blur.png")
                    logging.info("Saved 'debug_ocr_sample_input_for_blur.png'")
                except Exception as e_save_debug:
                    logging.warning(f"Could not save debug_ocr_sample_input_for_blur.png: {e_save_debug}")

            ocr_data = pytesseract.image_to_data(frame_pil_local, output_type=pytesseract.Output.DICT)
            logging.debug(f"OCR Data for blur function: {ocr_data}") # Use logging

        except pytesseract.TesseractNotFoundError:
            logging.error(
                "BLUR_FUNC: TESSERACT NOT FOUND. On Streamlit Cloud, ensure 'tesseract-ocr' "
                "is in packages.txt. Locally, ensure Tesseract is installed and in PATH, "
                "or provide tesseract_cmd_path."
            )
            logging.error(f"BLUR_FUNC: Pytesseract's current tesseract_cmd: '{pytesseract.pytesseract.tesseract_cmd}'")
            return None
        except Exception as e:
            logging.error(f"BLUR_FUNC: An error occurred during Pytesseract OCR: {e}")
            return None

        n_boxes = len(ocr_data['level'])
        subtitle_word_boxes = []
        img_width, img_height = frame_pil_local.size
        expected_subtitle_y_start_pixel = img_height * expected_y_start_ratio_local

        for i in range(n_boxes):
            confidence = int(ocr_data['conf'][i])
            if confidence > min_confidence_local:
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                x, y, w, h = (
                    ocr_data['left'][i], ocr_data['top'][i],
                    ocr_data['width'][i], ocr_data['height'][i]
                )
                if (y + (h / 2)) > expected_subtitle_y_start_pixel:
                    subtitle_word_boxes.append((x, y, x + w, y + h))

        if not subtitle_word_boxes:
            logging.info("BLUR_FUNC: No subtitle-like text found for blurring.")
            return None

        min_x = min(b[0] for b in subtitle_word_boxes)
        min_y = min(b[1] for b in subtitle_word_boxes)
        max_x = max(b[2] for b in subtitle_word_boxes)
        max_y = max(b[3] for b in subtitle_word_boxes)

        final_bbox = (
            max(0, min_x - padding_local), max(0, min_y - padding_local),
            min(img_width, max_x + padding_local), min(img_height, max_y + padding_local)
        )
        logging.info(f"BLUR_FUNC: Determined subtitle bounding box: {final_bbox}")
        return final_bbox

    # --- Nested Helper Function: Blur a Region in a Frame ---
    def _blur_region_in_frame(frame_np_local, bbox_local, kernel_local):
        if bbox_local is None:
            return frame_np_local
        x1, y1, x2, y2 = [int(c) for c in bbox_local]
        h_img, w_img = frame_np_local.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x1 >= x2 or y1 >= y2: return frame_np_local
        region_to_blur = frame_np_local[y1:y2, x1:x2]
        if region_to_blur.size == 0: return frame_np_local

        blurred_region = cv2.GaussianBlur(region_to_blur, kernel_local, 0)
        output_frame = frame_np_local.copy()
        output_frame[y1:y2, x1:x2] = blurred_region
        return output_frame

    # --- Main logic of blur_subtitles_in_video_unified ---
    if not os.path.exists(video_path):
        logging.error(f"BLUR_FUNC: Input video not found: '{video_path}'")
        return False # Indicate failure

    if video_path == output_path:
        logging.error(f"BLUR_FUNC: Input path and output path cannot be the same: '{video_path}'")
        return False # Indicate failure

    original_pytesseract_cmd = pytesseract.pytesseract.tesseract_cmd
    tesseract_path_was_set = False
    if tesseract_cmd_path and tesseract_cmd_path.strip():
        if os.path.exists(tesseract_cmd_path) and os.access(tesseract_cmd_path, os.X_OK):
            try:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
                logging.info(f"BLUR_FUNC: Pytesseract using explicit Tesseract path: {tesseract_cmd_path}")
                tesseract_path_was_set = True
            except Exception as e_tess_path:
                logging.warning(f"BLUR_FUNC: Failed to set Tesseract path to '{tesseract_cmd_path}': {e_tess_path}.")
        else:
            logging.warning(f"BLUR_FUNC: Provided Tesseract path '{tesseract_cmd_path}' not executable.")
    else:
        logging.info("BLUR_FUNC: No explicit Tesseract path. Using system PATH.")

    clip = None
    processed_clip = None
    try:
        logging.info(f"BLUR_FUNC: Loading video: {video_path}")
        clip = VideoFileClip(video_path)

        # Ensure sample_time_sec is within video duration
        valid_sample_time_sec = clip.duration - 0.1 if clip.duration > 0 else 0
        sample_time_sec = min(sample_time_sec, valid_sample_time_sec)
        sample_time_sec = max(0, sample_time_sec) # ensure non-negative

        logging.info(f"BLUR_FUNC: Extracting sample frame at {sample_time_sec:.2f}s...")
        sample_frame_np = clip.get_frame(sample_time_sec)
        sample_frame_pil = Image.fromarray(sample_frame_np)

        if debug_save_frames:
            try:
                sample_frame_pil.save("debug_blur_sample_frame.png")
                logging.info("BLUR_FUNC: Saved 'debug_blur_sample_frame.png'")
            except Exception as e_save:
                logging.warning(f"BLUR_FUNC: Could not save debug_blur_sample_frame.png: {e_save}")
        
        logging.info("BLUR_FUNC: Determining subtitle bounding box...")
        subtitle_bbox = _determine_subtitle_bbox_from_frame(
            sample_frame_pil, ocr_min_confidence, ocr_y_start_ratio, ocr_padding
        )

        if subtitle_bbox is None:
            logging.warning("BLUR_FUNC: Could not determine subtitle bbox. Blurring will not be applied.")
            # If no bbox, we might just copy the original to output_path or return indication
            # For now, let's assume we want to output the original if no blur region found.
            # Or, simply return False and let caller handle it.
            # A robust way is to copy the file if no processing happens:
            # import shutil
            # shutil.copy(video_path, output_path)
            # logging.info(f"BLUR_FUNC: Copied original video to {output_path} as no blur region found.")
            # return True # Indicate success (copying is success in this context)
            # --- OR ---
            return False # Indicate that blurring did not happen, let caller decide.

        if debug_save_frames:
            try:
                frame_viz = np.array(sample_frame_pil.copy())
                cv2.rectangle(frame_viz,
                              (int(subtitle_bbox[0]), int(subtitle_bbox[1])),
                              (int(subtitle_bbox[2]), int(subtitle_bbox[3])),
                              (0, 255, 0), 3)
                Image.fromarray(frame_viz).save("debug_blur_sample_frame_with_bbox.png")
                logging.info("BLUR_FUNC: Saved 'debug_blur_sample_frame_with_bbox.png'.")
            except Exception as e_save_bbox:
                logging.warning(f"BLUR_FUNC: Could not save debug_blur_sample_frame_with_bbox.png: {e_save_bbox}")

        def _process_frame_for_moviepy(frame_np_moviepy):
            return _blur_region_in_frame(frame_np_moviepy, subtitle_bbox, blur_kernel_size)

        logging.info("BLUR_FUNC: Applying blur to video frames...")
        processed_clip = clip.fl_image(_process_frame_for_moviepy)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir) # Ensure output directory exists

        logging.info(f"BLUR_FUNC: Writing blurred video to: {output_path}")
        
        # Optimized write_videofile call
        ffmpeg_params_list = ['-movflags', 'faststart'] # Basic for web
        output_fps = clip.fps if clip.fps and clip.fps > 0 else 24
        num_threads = os.cpu_count() or 2 # Sensible default

        processed_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac", # Retains original audio or adds silent if none
            threads=num_threads,
            fps=output_fps,
            preset='medium', # Balance between speed and quality
            logger='bar', # Or None
            ffmpeg_params=ffmpeg_params_list
        )
        logging.info(f"BLUR_FUNC: Successfully processed and saved blurred video to: {output_path}")
        return True # Indicate success

    except Exception as e:
        logging.error(f"BLUR_FUNC: Error during video blurring for '{video_path}': {e}", exc_info=True)
        return False # Indicate failure
    finally:
        if clip:
            try: clip.close()
            except: pass
        if processed_clip:
            try: processed_clip.close()
            except: pass
        if tesseract_path_was_set:
            pytesseract.pytesseract.tesseract_cmd = original_pytesseract_cmd
            logging.info("BLUR_FUNC: Restored original Pytesseract command path.")



def create_topic_summary_dataframe(selected_videos_dict, input_df=None):
    """
    Creates a DataFrame summarizing generated videos grouped by a normalized
    'topic_language' key. When an input table is provided, ensure that every
    topic row (even ones requesting a count of zero) appears in the same order
    with blank URL slots.
    """

    def _normalize_topic_lang(topic_val, lang_val):
        topic_clean = str(topic_val or '').strip().lower()
        lang_clean = str(lang_val or '').strip().lower()
        if not topic_clean or not lang_clean:
            return None
        return f"{topic_clean}_{lang_clean}"

    topic_lang_to_generated_urls = {}

    # 1. Collect Generated URLs and Group by Normalized Topic + Language
    for job_key, video_data in selected_videos_dict.items(): # Iterate through job entries
        grouping_key = _normalize_topic_lang(
            video_data.get('Topic', ''),
            video_data.get('Language', '')
        )
        s3_url = video_data.get('Generated S3 URL')

        if grouping_key and s3_url:
            topic_lang_to_generated_urls.setdefault(grouping_key, []).append(s3_url)

    input_df_valid = isinstance(input_df, pd.DataFrame) and not input_df.empty
    if not topic_lang_to_generated_urls and not input_df_valid:
        return pd.DataFrame(columns=['Topic'])

    # 2. Determine Max URLs per Group and Prepare Data
    max_urls = max(
        (len(urls) for urls in topic_lang_to_generated_urls.values()),
        default=0
    )

    def _build_row_for_key(topic_lang_key):
        row = {'Topic': topic_lang_key}
        urls = topic_lang_to_generated_urls.get(topic_lang_key, [])
        padded_urls = urls + [''] * max(0, max_urls - len(urls))
        for i, url in enumerate(padded_urls):
            row[f'vid{i+1}_url'] = url
        return row

    data_for_df = []
    seen_keys = set()

    if input_df_valid:
        for _, row in input_df.iterrows():
            grouping_key = _normalize_topic_lang(row.get('Topic', ''), row.get('Language', ''))
            if not grouping_key or grouping_key in seen_keys:
                continue
            seen_keys.add(grouping_key)
            data_for_df.append(_build_row_for_key(grouping_key))

    for topic_lang_key in topic_lang_to_generated_urls.keys():
        if topic_lang_key in seen_keys:
            continue
        data_for_df.append(_build_row_for_key(topic_lang_key))

    # 3. Create Final DataFrame
    if data_for_df:
        df_final = pd.DataFrame(data_for_df)
        if 'Topic' in df_final.columns:
            topic_col = df_final.pop('Topic')
            df_final.insert(0, 'Topic', topic_col)
            url_cols_present = [col for col in df_final.columns if col.startswith('vid')]
            url_cols_sorted = sorted(
                url_cols_present,
                key=lambda x: int(x.replace('vid', '').replace('_url', ''))
            )
            df_final = df_final[['Topic'] + url_cols_sorted]
        else:
            df_final = pd.DataFrame(columns=['Topic'])
    else:
        df_final = pd.DataFrame(columns=['Topic'])

    return df_final

# --- Helper Function: YouTube API Search ---
def search_youtube(api_key, query, max_results_per_term=5,max_retries = 5):
    """
    Performs a Youtube using the v3 API.
    Handles multiple terms generated by GPT or separated by '|'.
    Args:
        api_key (str): YouTube Data API v3 key.
        query (str): The search query string (can contain '|' or be generated).
        max_results_per_term (int): Max results to request for EACH term.
    Returns:
        list: A list of video dictionaries [{'title': '', 'videoId': '', 'url': ''}]
              or None if a critical API error occurs.
    """
    videos_res = []
    # Split query into multiple terms if it contains '|' or handle 'auto' case (which generates '|' separated terms)
    if '|' in query:
        # Clean up potential extra quotes from GPT generation
        query = query.replace('"','').replace("'",'')
        terms = [term.strip() for term in query.split('|') if term.strip()]
        count = max_results_per_term//len(terms)
    else:
        terms = [query] # Treat as a single term
        count = max_results_per_term

    st.write(f"Searching for terms: {terms} (Max {count} results per term)")

    total_fetched = 0
    MAX_TOTAL_RESULTS = 100 # Overall limit across all terms for safety

    for term in terms:
        tries = 0
        flag = False
        # st.text(str(max_retries)+   str(tries ))
        # input()
        while max_retries > tries and not flag:
            api_key_choice = random.choice(api_key)
            
            # term = term.replace("|","%7C")
            if 1==2:
                st.warning(f"Reached overall result limit ({MAX_TOTAL_RESULTS}). Stopping search.")
                break

            params = {
                'part': 'snippet',
                'q': term,
                'key': api_key_choice,
                'type': 'video',
                'maxResults': count,
                # 'videoEmbeddable': 'true',
                # 'order': 'viewCount', # Default is relevance
                'regionCode': 'US', # Optional: Bias results towards a region
                'safeSearch' : 'none'
            }
            with st.status("Progress:"):
                try:
                    response = requests.get(YOUTUBE_API_BASE_URL, params=params, timeout=15)
                    response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                    results = response.json()
            
                    processed_ids_this_term = set() # Avoid adding duplicates from the same term search

                    if 'items' in results:
                        for item in results['items']:
                            if total_fetched >= MAX_TOTAL_RESULTS: break # Check limit again

                            if item.get('id', {}).get('kind') == 'youtube#video' and 'videoId' in item['id']:
                                video_id = item['id']['videoId']
                                # Check if we already added this video ID from this specific term search
                                if video_id in processed_ids_this_term:
                                    continue

                                title = item['snippet'].get('title', 'No Title')
                                # Use the standard embeddable URL format
                                standard_url = f"https://www.youtube.com/watch?v={video_id}"
                                videos_res.append({
                                    'title': title,
                                    'videoId': video_id,
                                    'url': standard_url, # Store the standard watch URL
                                    'platform': 'yt'
                                    # 'thumbnail' : thumbnail_url
                                })
                                processed_ids_this_term.add(video_id)
                                total_fetched += 1
                                flag =True
                                
                except requests.exceptions.Timeout:
                    st.text(f"API Request Timeout for query '{term}'.")
                    
                except requests.exceptions.HTTPError as http_err:
                    st.text(f"API HTTP Error for query '{term}': {http_err}")
                    
                    # Check for common quota/key errors
                    if response.status_code == 403:
                        st.text("Received 403 Forbidden. Check your YouTube API Key and Quota.")
                        
                        # return None # Signal critical error
                    if response.status_code == 400:
                        st.text(f"Received 400 Bad Request. Check API parameters. Details: {response.text}")
                        

                except requests.exceptions.RequestException as e:
                    st.text(f"API Request Error for query '{term}': {e}")
                    
                except Exception as e:
                    st.text(f"An unexpected error occurred during search for '{term}': {e}")
                    import traceback
                    st.text(traceback.format_exc())
                finally:
                    tries += 1


    # Return collected results, respecting the overall MAX_TOTAL_RESULTS implicitly
    # No need to slice here as the loop breaks early
    return videos_res


# def search_tiktok_links_google(api_keys, cx_id, query, num_results=20, max_retries=3):
    # import requests, time, random

    # max_per_page = 10
    # query_terms = query.split("|")
    # num_results = min(num_results, 100)
    # video_links_info = []
    # total_collected = 0
    # term_cycle = query_terms * ((num_results // len(query_terms)) + 1)  # cycle terms to reach total target

    # for term in term_cycle:
    #     if total_collected >= num_results:
    #         break

    #     search_query = f"site:tiktok.com inurl:/video/ {term.replace('#','').replace('shorts','').replace("'", '')}"
    #     collected_for_term = 0

    #     for start in range(1, 100, max_per_page):  # CSE allows up to start=91
    #         if total_collected >= num_results:
    #             break

    #         tries = 0
    #         while tries < max_retries:
    #             try:
    #                 api_key = random.choice(api_keys)
    #                 params = {
    #                     'key': api_key,
    #                     'cx': cx_id,
    #                     'q': search_query,
    #                     'num': min(max_per_page, num_results - total_collected),
    #                     'start': start,
    #                     'searchType': 'image',
    #                     'gl': 'us'
    #                 }

    #                 response = requests.get("https://customsearch.googleapis.com/customsearch/v1", params=params, timeout=10)
    #                 response.raise_for_status()
    #                 results_data = response.json()

    #                 if 'items' in results_data:
    #                     for item in results_data['items']:
    #                         url = item['image'].get("contextLink", "")
    #                         if 'video' not in url:
    #                             continue

    #                         video_id = url.split("/")[-1]
    #                         title = item.get("title", "")
    #                         thumbnail_url = item.get("link", "")

    #                         video_links_info.append({
    #                             'title': title,
    #                             'url': url,
    #                             'thumbnail_url': thumbnail_url,
    #                             'videoId': video_id,
    #                             'platform': 'tk'
    #                         })

    #                         total_collected += 1
    #                         collected_for_term += 1

    #                 break

    #             except Exception as e:
    #                 logging.info(f"[Retry {tries+1}] Error: {e}")
    #                 tries += 1
    #                 if tries < max_retries:
    #                     time.sleep(1)

    #         if collected_for_term == 0:
    #             break  # this term exhausted

    # return video_links_info
def search_tiktok_links_google(api_keys, cx_id, query, num_results=20, start_index=1, max_retries=6):
    """
    Searches for TikTok video pages using Google Custom Search API, supporting pagination for more than 10 results.
    Args:
        api_key (str): Google API Key.
        cx_id (str): Custom Search Engine ID.
        query (str): Search query.
        num_results (int): Total number of results to return (max 100).
        max_retries (int): Retry count on errors.
    Returns:
        list: List of video dictionaries or None if critical error.
    """
    import requests
    import time
    from urllib.parse import urlencode

    if isinstance(api_keys, str):
        api_keys = [api_keys]
    elif isinstance(api_keys, dict):
        api_keys = list(api_keys.values())
    elif isinstance(api_keys, set):
        api_keys = list(api_keys)
    api_keys = [key for key in api_keys if key]
    if not api_keys:
        st.error("No Google API keys available for TikTok search.", icon="⚠️")
        return ([], None)

    search_query_on_google = f"{query.replace('#','').replace('shorts','')} site:www.tiktok.com/@"
    max_per_page = 10  # Google Custom Search API limit for 'num' is 10
    all_fetched_videos = []
    current_start_offset = start_index
    total_videos_fetched_across_pages = 0 # Tracks videos fetched across all internal pages for THIS call

    st.write(f"\nSearching Google for TikTok links with: '{search_query_on_google}', wanting {num_results} starting from {start_index}")

    while total_videos_fetched_across_pages < num_results:
        num_to_fetch_this_page = min(max_per_page, num_results - total_videos_fetched_across_pages)
        if num_to_fetch_this_page <= 0:
            break # Already fetched enough or invalid request

        tries = 0
        page_videos = []
        api_call_succeeded = False
        
        while tries < max_retries:
            try:
                api_key = random.choice(api_keys)
                params = {
                    'key': api_key,
                    'cx': cx_id,
                    'q': search_query_on_google,
                    'num': num_to_fetch_this_page,
                    'start': current_start_offset,
                    'searchType': 'image',
                    'gl': 'us'
                }
                # st.write(f"API Call Params: {params}") # Debugging

                response = requests.get("https://customsearch.googleapis.com/customsearch/v1", params=params, timeout=15)
                response.raise_for_status()
                results_data = response.json()
                
                items_on_page = results_data.get('items', [])
                for item in items_on_page:
                    title = item.get("title", "")
                    url = item['image'].get("contextLink", "")
                    if 'video' not in url: # Ensure it's a video link
                        continue
                    video_id = url.split("/")[-1]
                    thumbnail_url = item['image'].get("thumbnailLink", "")
                    page_videos.append({
                        'title': title, 'url': url, 'thumbnail_url': thumbnail_url,
                        'videoId': video_id, 'platform': 'tk'
                    })
                
                all_fetched_videos.extend(page_videos)
                total_videos_fetched_across_pages = len(all_fetched_videos)
                api_call_succeeded = True
                # st.write(f"API call success: Fetched {len(page_videos)} items for start {current_start_offset}. Total now: {total_videos_fetched_across_pages}") # Debugging
                break  # Success for this page, break retry loop

            except requests.exceptions.RequestException as e:
                st.warning(f"[Attempt {tries+1}/{max_retries}] Request error for start {current_start_offset}: {e}")
            except Exception as e:
                st.error(f"Unexpected error for start {current_start_offset}: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            tries += 1
            if tries < max_retries:
                time.sleep(1) # Wait before retrying

        if not api_call_succeeded:
            st.error(f"Failed to fetch data for start_index {current_start_offset} after {max_retries} retries.")
            # Depending on desired strictness, you could 'return ([], None)' here to indicate partial failure
            # For now, we'll continue and return what we have, but next_page_start_index will be None.
            return (all_fetched_videos, None) # Critical failure for a page

        if len(page_videos) < num_to_fetch_this_page:
            # API returned fewer items than requested for this page, so it's the last page.
            # st.write(f"API returned fewer items ({len(page_videos)}) than requested ({num_to_fetch_this_page}). Assuming end of results.") # Debugging
            break # Break from the outer while loop (fetching pages)
        
        # Standard increment for Google CSE pagination for `start` is by the number of items requested for that page.
        current_start_offset += num_to_fetch_this_page 

        if total_videos_fetched_across_pages >= num_results:
            break # Fetched enough videos as requested by num_results

        time.sleep(0.1) # Small delay between calls to be polite to the API

    next_page_start_index = None
    if total_videos_fetched_across_pages >= num_results : # If we fulfilled or exceeded the request
        # Check if the *last page fetched* was full. If it was, there *might* be more.
        # This logic is tricky because we don't know if the API has more unless we query.
        # A simpler approach: if we fetched *exactly* num_results, and the last page was full,
        # then it's likely there are more.
        # However, the prompt asks: "if num_results videos were successfully fetched), next_page_start_index = start_index + len(all_fetched_videos)"
        # This implies next_page_start_index is based on the original start_index + total collected.
        next_page_start_index = start_index + total_videos_fetched_across_pages
    # else: next_page_start_index remains None (API exhausted before num_results)
    
    # st.write(f"Returning: {len(all_fetched_videos)} videos, next_page_start_index: {next_page_start_index}") # Debugging
    return (all_fetched_videos, next_page_start_index)

def search_pinterest_links_google(api_keys, cx_id, query, num_results=20, start_index=1, max_retries=6):
    """
    Searches for Pinterest pins (video ideas) via Google Custom Search, mirroring the TikTok workflow.
    Returns (videos_list, next_page_start_index).
    """
    import requests
    import time

    if isinstance(api_keys, str):
        api_keys = [api_keys]
    elif isinstance(api_keys, dict):
        api_keys = list(api_keys.values())
    elif isinstance(api_keys, set):
        api_keys = list(api_keys)
    api_keys = [key for key in api_keys if key]
    if not api_keys:
        st.error("No Google API keys available for Pinterest search.", icon="⚠️")
        return ([], None)

    search_query_on_google = f"""{query.replace('#','').replace('shorts','')} site:www.pinterest.com/pin/ videos/thumbnails """
    max_per_page = 10
    all_fetched_videos = []
    current_start_offset = start_index
    total_videos_fetched_across_pages = 0
    next_page_start_index = None

    st.write(f"\nSearching Google for Pinterest pins with: '{search_query_on_google}', wanting {num_results} starting from {start_index}")

    while total_videos_fetched_across_pages < num_results:
        num_to_fetch_this_page = min(max_per_page, num_results - total_videos_fetched_across_pages)
        if num_to_fetch_this_page <= 0:
            break

        tries = 0
        page_videos = []
        api_call_succeeded = False

        while tries < max_retries:
            try:
                api_key = random.choice(api_keys)
                params = {
                    'key': api_key,
                    'cx': cx_id,
                    'q': search_query_on_google,
                    'num': num_to_fetch_this_page,
                    'start': current_start_offset,
                    'searchType': 'image',
                    'gl': 'us'
                }

                response = requests.get("https://customsearch.googleapis.com/customsearch/v1", params=params, timeout=15)
                response.raise_for_status()
                results_data = response.json()

                items_on_page = results_data.get('items', [])
                for item in items_on_page:
                    image_block = item.get('image', {})
                    context_link = image_block.get("contextLink") or item.get("link", "")
                    if "/pin/" not in context_link:
                        continue
                    pin_id = extract_pinterest_pin_id(context_link)
                    if not pin_id:
                        continue
                    thumbnail_url = image_block.get("thumbnailLink") or item.get("link", "")
                    title = item.get("title", "")
                    page_videos.append({
                        'title': title,
                        'url': context_link,
                        'thumbnail_url': thumbnail_url,
                        'videoId': pin_id,
                        'platform': 'pin'
                    })

                all_fetched_videos.extend(page_videos)
                total_videos_fetched_across_pages = len(all_fetched_videos)
                api_call_succeeded = True
                break

            except requests.exceptions.RequestException as e:
                st.warning(f"[Attempt {tries+1}/{max_retries}] Request error for Pinterest start {current_start_offset}: {e}")
            except Exception as e:
                st.error(f"Unexpected error for Pinterest start {current_start_offset}: {e}")
                import traceback
                st.error(traceback.format_exc())

            tries += 1
            if tries < max_retries:
                time.sleep(1)

        if not api_call_succeeded:
            st.error(f"Failed to fetch data for Pinterest start_index {current_start_offset} after {max_retries} retries.")
            return (all_fetched_videos, None)

        if len(page_videos) < num_to_fetch_this_page:
            next_page_start_index = None
            break

        current_start_offset += num_to_fetch_this_page

    else:
        next_page_start_index = start_index + total_videos_fetched_across_pages

    return (all_fetched_videos, next_page_start_index)
# --- Helper Function: Simple Hash ---
def simple_hash(s):
    """Creates a simple, short hash string from an input string for UI keys."""
    total = 0
    for i, c in enumerate(s):
        total += (i + 1) * ord(c)
    return str(total % 100000) # Keep it relatively short

def extract_pinterest_pin_id(url):
    """Extracts the Pinterest pin ID from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        segments = [seg for seg in parsed.path.split('/') if seg]
        for idx, seg in enumerate(segments):
            if seg.lower() == 'pin' and idx + 1 < len(segments):
                candidate = segments[idx + 1]
                break
        else:
            candidate = segments[-1] if segments else ""
        return candidate.split('?')[0].split('&')[0]
    except Exception:
        return None


# --- Helper Function: Get Info with yt-dlp ---
def get_yt_dlp_info(video_url):
    """
    Uses yt-dlp to extract video format information, prioritizing direct mp4 URLs.
    Returns a dictionary: {'direct_url': url, 'format_details': str, 'error': None_or_str}
    """
    # Prioritize standard mp4 formats often playable directly
    # Format 18: mp4 [360p], Format 22: mp4 [720p]
    # Fallback to best mp4, then best overall
    YDL_OPTS = {
        'format': 'bestvideo[ext=mp4]',   #        'format': '22/18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'skip_download': True, # Don't download, just get info
        'extract_flat': False, # Need format details if direct URL isn't top-level
        'socket_timeout': YT_DLP_FETCH_TIMEOUT,
        'retries': 3, # Add retries for network issues
        'fragment_retries': 3,
        # Uncomment if using cookies and path is valid
        # 'cookiefile': COOKIE_FILE_PATH if COOKIE_FILE_PATH and os.path.exists(COOKIE_FILE_PATH) else None,
    }
    if 'tiktok' in video_url:
        YDL_OPTS['format'] = 'best'
    # Add cookiefile only if path exists
    if COOKIE_FILE_PATH and os.path.exists(COOKIE_FILE_PATH):
        YDL_OPTS['cookiefile'] = COOKIE_FILE_PATH
        logging.info(f"yt-dlp: Using cookie file: {COOKIE_FILE_PATH}")
    elif COOKIE_FILE_PATH:
        logging.info(f"yt-dlp: Cookie file path provided but not found: {COOKIE_FILE_PATH}")


    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            # Extract info. This is the network call.
            info = ydl.extract_info(video_url, download=False)

            # --- Extract relevant details ---
            direct_url = info.get('url') # yt-dlp often puts the best direct URL here based on 'format'
            format_note = info.get('format_note', 'N/A')
            format_id = info.get('format_id', 'N/A') # The ID yt-dlp selected
            ext = info.get('ext', 'N/A')
            resolution = info.get('resolution', info.get('width') and info.get('height') and f"{info['width']}x{info['height']}") or 'N/A'
            filesize = info.get('filesize') or info.get('filesize_approx')

            # --- Fallback: Check 'formats' list if top-level 'url' is missing ---
            # This can happen for some sites or if specific format selection failed slightly differently
            if not direct_url and 'formats' in info:
                logging.info(f"yt-dlp: Top-level URL missing for {video_url}. Checking formats list...")
                selected_format = None
                # Check the format yt-dlp claims it selected first
                if format_id != 'N/A':
                    selected_format = next((f for f in info['formats'] if f.get('format_id') == format_id and f.get('url')), None)

                # If that didn't work, try preferred formats manually
                if not selected_format:
                    preferred_formats = ['22', '18']
                    for pf_id in preferred_formats:
                        selected_format = next((f for f in info['formats'] if f.get('format_id') == pf_id and f.get('url')), None)
                        if selected_format: break

                # If still no URL, grab the *last* format in the list with a URL (often the 'best')
                # Or maybe the first one? Let's try the last one yt-dlp considered.
                if not selected_format:
                     selected_format = next((f for f in reversed(info['formats']) if f.get('url')), None)


                if selected_format:
                    logging.info(f"yt-dlp: Found URL in formats list (Format ID: {selected_format.get('format_id')})")
                    direct_url = selected_format.get('url')
                    # Update details based on the found format
                    format_id = selected_format.get('format_id', format_id)
                    format_note = selected_format.get('format_note', format_note)
                    ext = selected_format.get('ext', ext)
                    resolution = selected_format.get('resolution', resolution)
                    filesize = selected_format.get('filesize') or selected_format.get('filesize_approx') or filesize

            # --- Format details string ---
            filesize_str = f"{filesize / (1024*1024):.2f} MB" if filesize else "N/A"
            format_details = f"ID: {format_id}, Res: {resolution}, Ext: {ext}, Size: {filesize_str}"

            # --- Return result ---
            if direct_url:
                # st.text({
                #     'direct_url': direct_url,
                #     'format_details': format_details,
                #     'error': None
                # })
                return {
                    'direct_url': direct_url,
                    'format_details': format_details,
                    'error': None
                }
            else:
                # Log available info if URL extraction failed unexpectedly
                logging.info(f"Warning: Could not extract direct URL for {video_url} even after checking formats. Info keys: {info.keys()}")
                # Try to find a reason
                reason = "Unknown reason"
                if info.get('availability'): reason = f"Availability: {info['availability']}"
                if info.get('_type') == 'playlist': reason = "Cannot process playlist URL directly"
                return {'direct_url': None, 'format_details': 'Error', 'error': f'Could not extract direct video URL. ({reason})'}

    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e)
        # Simplify common error messages
        if "confirm your age" in err_msg.lower():
            error_reason = 'Age-restricted video'
        elif "private video" in err_msg.lower():
            error_reason = 'Private video'
        elif "video unavailable" in err_msg.lower():
            error_reason = 'Video unavailable'
        elif "login required" in err_msg.lower():
             error_reason = 'Login required (try using cookies)'
        else:
             error_reason = f"yt-dlp: {err_msg[:150]}" # Truncate long messages
        st.warning(f"yt-dlp DownloadError for {video_url}: {error_reason}", icon="🚧")
        return {'direct_url': None, 'format_details': 'Error', 'error': error_reason}
    except Exception as e:
        st.error(f"Unexpected yt-dlp error for {video_url}: {e}", icon="💥")
        import traceback
        st.error(traceback.format_exc())
        return {'direct_url': None, 'format_details': 'Error', 'error': f"Unexpected yt-dlp error: {e}"}

def download_vid_ytdlp(video_url, output_dir="downloads", filename_template="%(title).80s.%(ext)s"):
    """
    Downloads a video using yt-dlp and returns the path to the downloaded file.
    Args:
        video_url (str): The video URL to download.
        output_dir (str): Directory where the video will be saved.
        filename_template (str): Template for the output filename.

    Returns:
        str: Path to the downloaded file, or None if failed.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path_template = os.path.join(output_dir, filename_template)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path_template,
        'quiet': True,
        'noplaylist': True,
        'merge_output_format': 'mp4',  # ensures muxing into a single MP4 if needed
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            # If merged into mp4, manually fix extension
            if 'ext' in info and info['ext'] != 'mp4':
                downloaded_file = os.path.splitext(downloaded_file)[0] + '.mp4'
            return downloaded_file
    except yt_dlp.utils.DownloadError as e:
        logging.info(f"[yt-dlp error] {e}")
        return None
    except Exception as e:
        logging.info(f"[unexpected error] {e}")
        return None
# --- Helper Function: Generate Script with ChatGPT ---

def gemini_text_lib(prompt,model ='gemini-2.5-pro-exp-03-25', is_with_file=False,file_url = None ):
    # if is_pd_policy : prompt += predict_policy




    client = genai.Client(api_key=random.choice(GEMINI_API_KEY))


    try:
        if is_with_file:
            file_extension ='jpg'
            with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=file_extension or '.tmp') as temp_file:
                st.text(file_url)
                res = requests.get(file_url)
                res.raise_for_status()  
                temp_file.write(res.content)
                
                file = client.files.upload(file=temp_file.name, config={'mime_type' :'image/jpeg'})
                response = client.models.generate_content(
                    model=model, contents=  [prompt, file]

                )
        elif not is_with_file:
            response = client.models.generate_content(
                model=model, contents=  prompt
            )

        return response.text
    except Exception as e:
        st.text('gemini_text_lib error ' + str(e))
        time.sleep(4)
        return None




def chatGPT(prompt, client, model="gpt-4o", temperature=1.0):
    """Generates text using OpenAI Chat Completion."""
    try:
        # Use the passed client object
        response = client.chat.completions.create(
             model=model,
             temperature=temperature,
             messages=[{'role': 'user', 'content': prompt}]
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        st.error(f"Error calling OpenAI (Model: {model}): {e}", icon="🤖")
        # Consider logging the full error: logging.info(f"OpenAI Error: {e}")
        return None


# --- Helper Function: Generate Script with Claude ---
def claude(prompt , model = "claude-sonnet-4-0", temperature=1 , is_thinking = False, max_retries = 10):
    tries = 0

    while tries < max_retries:
        try:
        
        
        
            client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=st.secrets["ANTHROPIC_API_KEY"])
        
            if is_thinking == False:
                    
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                
                top_p= 0.8,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
                return message.content[0].text
            if is_thinking == True:
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                thinking = { "type": "enabled",
                "budget_tokens": 16000}
            )
                return message.content[1].text
        
        
        
            logging.info(message)
            return message.content[0].text

        except Exception as e:
            st.text(e)
            tries += 1 
            time.sleep(5)



# --- Helper Function: Generate TTS Audio & Timestamps ---
def generate_audio_with_timestamps(text, client, voice_id="sage"):
    """Generates TTS audio using OpenAI, saves it, gets word timestamps via Whisper."""
    temp_audio_path = None
    word_timings = [] # Ensure initialized

    try:
        if not text or not text.strip():
            raise ValueError("Input text for TTS cannot be empty.")

        instructions_per_voice ={
            'redneck': {'instructions':'talk like an older ameircan redneck heavy accent. deep voice, enthusiastic','voice' :'ash'},
            'announcer': {'instructions':'Polished announcer voice, American accent','voice' :'ash'},
            'sage uk': {'instructions':'Polished announcer voice, British accent','voice' :'sage'},
            'sage': {'instructions': 'high energy enthusiastic', 'voice': 'sage'},
            'announcer uk': {'instructions':'Polished announcer voice, British accent','voice' :'ash'}



        }
        if voice_id in instructions_per_voice.keys():
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts", # Use HD for better quality
                voice=instructions_per_voice[voice_id]['voice'],
                input=text,
                response_format="mp3",
                instructions= instructions_per_voice[voice_id]['instructions'],
                speed=1.15
            )

        elif voice_id in['sage']:
        # 1. Generate TTS audio
            response = client.audio.speech.create(
                model="tts-1-hd", # Use HD for better quality
                voice=voice_id,
                input=text,
                response_format="mp3",
                speed=1.0
            )

        # 2. Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file_obj:
            temp_audio_path = temp_audio_file_obj.name
            temp_audio_file_obj.write(response.content)

        # 3. Optional: Volume Boost using pydub
        try:
            boosted_audio = AudioSegment.from_file(temp_audio_path)
            # Adjust dB boost as needed (+6 dB approx doubles loudness)
            boosted_audio = boosted_audio + 8 # Boost by 8 dB
            boosted_audio.export(temp_audio_path, format="mp3")
            # logging.info(f"Audio volume boosted for {temp_audio_path}") # Debug log
        except Exception as boost_err:
            st.warning(f"Could not boost audio volume: {boost_err}", icon="🔊")

        # 4. Transcribe boosted audio with Whisper for timestamps
        with open(temp_audio_path, "rb") as audio_file_rb:
            # Use the client passed to the function
            transcribe_response = client.audio.transcriptions.create(
                file=audio_file_rb,
                model="whisper-1", # Use appropriate Whisper model
                response_format="verbose_json", # Needed for timestamps
                timestamp_granularities=["word"] # Request word-level timestamps
            )

        # 5. Process Transcription Response (Handles SDK v1.x+ response object)
        if hasattr(transcribe_response, 'words') and transcribe_response.words:
            for word_info in transcribe_response.words:
                start_time = getattr(word_info, 'start', None)
                end_time = getattr(word_info, 'end', None)
                # Validate timestamps before adding
                if start_time is not None and end_time is not None:
                    try:
                        start_float = float(start_time)
                        end_float = float(end_time)
                        if end_float >= start_float: # Ensure end is not before start
                            word_timings.append({
                                "word": getattr(word_info, 'word', ''),
                                "start": start_float,
                                "end": end_float
                            })
                        else:
                             st.warning(f"Skipping invalid time range (end <= start) for word '{getattr(word_info, 'word', '')}': start={start_float}, end={end_float}", icon="⏱️")
                    except (ValueError, TypeError):
                         st.warning(f"Skipping invalid timestamp format for word '{getattr(word_info, 'word', '')}': start={start_time}, end={end_time}", icon="⏱️")
                else:
                    st.warning(f"Missing start/end time for word '{getattr(word_info, 'word', '')}'", icon="⏱️")
        else:
             st.warning("Whisper did not return word timestamps in the expected format.", icon="⏱️")
             # Log the response structure for debugging if necessary
             logging.info("Unexpected transcription response structure:", transcribe_response)


        # Return path and timings if successful
        if not word_timings:
             st.warning("No valid word timings extracted after transcription.", icon="⚠️")
             # Return None for timings to indicate failure downstream? Or empty list?
             # Let's return empty list but log the warning.

        # Return path even if timings are empty, but signal timings issue
        return temp_audio_path, word_timings

    except Exception as e:
        st.error(f"OpenAI API Error in TTS/Timestamp: {e}", icon="🤖")
        # last_error = api_err
  

    # --- Cleanup on Error ---
    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
            logging.info(f"Cleaned up temp audio file on error: {temp_audio_path}") # Debug log
        except Exception as rm_err:
            st.warning(f"Could not remove temp audio file {temp_audio_path} during error handling: {rm_err}")

    # Return None, None on failure
    return None, None


# --- Helper Function: Group Word Timings ---
def group_words_with_timing(word_timings, words_per_group=2):
    """Groups words and their timings for subtitles."""
    grouped_timings = []
    if not word_timings: return grouped_timings # Handle empty input

    for i in range(0, len(word_timings), words_per_group):
        group_words_data = word_timings[i:i+words_per_group]
        if group_words_data:
            combined_word = " ".join(word_info['word'] for word_info in group_words_data).strip()
            # Ensure start/end times are valid before using
            if 'start' in group_words_data[0] and 'end' in group_words_data[-1]:
                 start_time = group_words_data[0]['start']
                 end_time = group_words_data[-1]['end']
                 # Basic validation
                 if combined_word and end_time > start_time:
                      grouped_timings.append({
                           "text": combined_word,
                           "start": start_time,
                           "end": end_time
                      })
                 # else: # Optional: Log skipped groups
                 #    logging.info(f"Skipping invalid subtitle group: {group_words_data}")
            # else: # Optional: Log missing start/end
            #    logging.info(f"Skipping group due to missing time data: {group_words_data}")

    return grouped_timings


# --- Helper Function: Create Text Image for Subtitles ---
def create_text_image(text, fontsize, color, bg_color, font_path, video_width):
    """
    Creates a transparent PNG image with text and rounded background,
    wrapping text to fit video width and centering it.
    """
    try:
        # --- Font Loading ---
        font = None
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except Exception as font_load_err:
                st.warning(f"Failed to load font {font_path}: {font_load_err}. Using default.", icon="⚠️")

        if not font: # If path was None or loading failed
            try:
                # Try getting default font with size (newer Pillow)
                font = ImageFont.load_default(size=fontsize)
            except AttributeError:
                # Fallback for older Pillow without size arg
                font = ImageFont.load_default()
                st.warning(f"Using basic default font. Consider providing a TTF font file.", icon="⚠️")


        # --- Configuration ---
        padding_x = 25  # Horizontal padding for the background
        padding_y = 15  # Vertical padding for the background
        bg_radius = 15  # Corner radius for the background
        # Calculate max width for the text itself inside the video frame
        max_text_width = video_width - (2 * padding_x) - 30
        if max_text_width <= 0: max_text_width = video_width // 2 # Safety net

        # --- Text Wrapping ---
        lines = []
        words = text.split()
        if not words:
            return np.zeros((10, 10, 4), dtype=np.uint8) # Handle empty text

        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Use getbbox for more accurate width/height, fallback to getlength
            try:
                 # bbox format: (left, top, right, bottom) relative to (0,0) anchor
                 bbox = font.getbbox(test_line)
                 line_width = bbox[2] - bbox[0]
            except AttributeError: # Fallback for older PIL/Pillow
                 line_width = font.getlength(test_line)


            if line_width <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Check if the single word itself is too long
                try:
                    word_bbox = font.getbbox(word)
                    word_width = word_bbox[2] - word_bbox[0]
                except AttributeError:
                    word_width = font.getlength(word)

                if word_width <= max_text_width:
                    current_line = word
                else:
                    # Word too long: Add previous line (if any), add long word on its own line
                    # This might still exceed width if the word is extremely long and cannot be broken
                    if current_line and lines[-1] != current_line: # Avoid double adding
                        pass # Already added above
                    lines.append(word) # Add the long word as its own line
                    current_line = "" # Reset

        if current_line: # Add the last line
            lines.append(current_line)

        wrapped_text = "\n".join(lines)
        if not wrapped_text: wrapped_text = text # Fallback

        # --- Calculate Text Block Dimensions ---
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        try:
            bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center') # Use center align here for bbox
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bbox_y_offset = bbox[1] # Top offset relative to draw origin
            text_width = max(text_width, 1)
            text_height = max(text_height, 1)
        except AttributeError:
            # Fallback
             st.warning("Using fallback subtitle dimension calculation (update Pillow recommended).", icon="PIL")
             text_width = 0
             for line in lines:
                  try:
                      line_bbox = font.getbbox(line)
                      text_width = max(text_width, line_bbox[2] - line_bbox[0])
                  except AttributeError:
                      text_width = max(text_width, int(font.getlength(line)))

             try:
                 ascent, descent = font.getmetrics()
                 line_height_metric = ascent + descent
             except AttributeError:
                 line_height_metric = fontsize * 1.2 # Estimate

             line_height = line_height_metric + 4 # Add spacing
             text_height = len(lines) * line_height
             bbox_y_offset = -int(fontsize * 0.1) # Rough guess
             text_width = max(text_width, 1)
             text_height = max(text_height, 1)

        # --- Create Final Image ---
        img_width = text_width + 2 * padding_x
        img_height = text_height + 2 * padding_y

        img = Image.new("RGBA", (int(img_width), int(img_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # --- Draw Background ---
        try:
            if isinstance(bg_color, str) and bg_color.startswith('rgba'):
                parts = bg_color.strip('rgba()').split(',')
                r, g, b = map(int, parts[:3])
                a = int(float(parts[3]) * 255)
                fill_color_tuple = (r, g, b, a)
            elif isinstance(bg_color, str): # hex or name
                 fill_color_tuple = bg_color # Pillow handles hex/names directly
            elif isinstance(bg_color, (tuple, list)) and len(bg_color) == 4:
                 fill_color_tuple = tuple(map(int, bg_color)) # Ensure integer tuple
            else: # assume tuple RGB, add alpha
                 fill_color_tuple = tuple(map(int, bg_color)) + (int(0.6 * 255),) if len(bg_color) == 3 else (0,0,0, int(0.6*255))

            draw.rounded_rectangle([(0, 0), (img_width, img_height)], radius=bg_radius, fill=fill_color_tuple)
        except Exception as draw_err:
             st.warning(f"Could not draw rounded rect: {draw_err}. Using simple rect.", icon="🎨")
             # Ensure fill_color_tuple is defined before drawing fallback rectangle
             if 'fill_color_tuple' not in locals():
                 fill_color_tuple = (0,0,0, int(0.6*255)) # Default fallback fill
             draw.rectangle([(0,0), (img_width, img_height)], fill=fill_color_tuple)

        # --- Draw Text ---
        text_x = padding_x
        text_y = padding_y - bbox_y_offset # Adjust vertical start based on calculated bbox top

        draw.multiline_text(
            (text_x, text_y),
            wrapped_text,
            font=font,
            fill=color,
            align="center",
            spacing=4,
            anchor="la" # Anchor at left-ascent of the first line
        )

        return np.array(img)

    except Exception as e:
        st.error(f"Error creating text image for '{text[:50]}...': {e}", icon="🎨")
        return np.zeros((10, 10, 4), dtype=np.uint8)




# --- Helper Function: Download with yt-dlp (Downloads the actual video file) ---
def download_with_ytdlp(video_url, cookie_file_path=None):
    """
    Uses yt-dlp to download a video to a local temp file.
    Args:
        video_url (str): The standard YouTube watch URL (or other yt-dlp supported URL).
        cookie_file_path (str, optional): Path to a Netscape format cookie file.
    Returns:
        str: Path to the downloaded temporary file, or None if download fails.
             Caller is responsible for deleting the file.
    """
    temp_path = None
    st.write(f"ℹ️ Attempting to download video content: {video_url}")
    if cookie_file_path and os.path.exists(cookie_file_path):
        st.write(f"ℹ️ Using cookie file: {cookie_file_path}")
    elif cookie_file_path:
         st.warning(f"⚠️ Cookie file specified but not found: {cookie_file_path}")


    try:
        # Create a temporary file for the download output
        # delete=False means we manage deletion manually
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_obj:
            temp_path = temp_file_obj.name

        # Configure yt-dlp options for downloading
        ydl_opts = {
            'outtmpl': temp_path, # Save download to this specific temp path
            'format': '22/18/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best', # Prefer reasonable quality mp4
            'quiet': False, # Show yt-dlp output in logs
            'noplaylist': True,
            'merge_output_format': 'mp4', # Ensure output is mp4 if merging is needed
            'overwrites': True, # Overwrite temp file if it somehow exists
            'retries': 3,
            'fragment_retries': 3,
            'socket_timeout': 60, # Longer timeout for download
            # Add cookie option if path is valid and file exists
            'cookiefile': cookie_file_path if cookie_file_path and os.path.exists(cookie_file_path) else None,
            # Add progress hook for potential Streamlit progress bar update (advanced)
            # 'progress_hooks': [lambda d: logging.info(d['status'], d.get('filename'))], # Example hook
        }

        st.write(f"⏳ Starting yt-dlp download to: {temp_path}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url]) # Pass the URL as a list

        # --- Basic Integrity Checks ---
        st.write("🔬 Performing basic checks on downloaded file...")
        if not os.path.exists(temp_path):
            st.error(f"❌ Download Error: File not found after yt-dlp finished: {temp_path}")
            return None # Return None if file doesn't exist
        if os.path.getsize(temp_path) == 0:
            st.error(f"❌ Download Error: File is empty after download: {temp_path}")
            try: os.remove(temp_path) # Clean up empty file
            except OSError: pass
            return None # Return None if file is empty

        file_size_mb = os.path.getsize(temp_path) / (1024*1024)
        st.write(f"✔️ Check Passed: File exists and is not empty (Size: {file_size_mb:.2f} MB).")

        st.success(f"✅ yt-dlp download successful: {temp_path}")
        return temp_path # Return the path to the downloaded file

    except yt_dlp.utils.DownloadError as dl_err:
        # Check for common download errors
        err_str = str(dl_err).lower()
        if "login" in err_str or "authentication" in err_str or "age restricted" in err_str or "private" in err_str:
            st.error(f"❌ yt-dlp DownloadError: Authentication possibly required or video is private/restricted. Cookies might help. Message: {dl_err}", icon="🔒")
        elif "403 forbidden" in err_str:
            st.error(f"❌ yt-dlp DownloadError: Received '403 Forbidden'. YouTube may be blocking the request. Cookies might help, or the video might be unavailable. Message: {dl_err}", icon="🚫")
        elif "video unavailable" in err_str:
             st.error(f"❌ yt-dlp DownloadError: Video unavailable. Message: {dl_err}", icon="❓")
        else:
            st.error(f"❌ yt-dlp DownloadError: {dl_err}")

        # Clean up potentially incomplete file
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass
        return None # Return None on download error
    except Exception as e:
        st.error(f"❌ Unexpected error during download_with_ytdlp: {e}")
        import traceback
        st.error(traceback.format_exc())
        # Clean up temp file on unexpected error
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass
        return None # Return None on unexpected error


# --- Helper Function: Download Direct URL (Less likely needed now, but kept for utility) ---
def download_direct_url(url, suffix=".mp4"):
    """Downloads content from a direct URL to a temporary local file."""
    local_path = None
    try:
        headers = { # Use a common user agent
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            local_path = temp_file.name
            logging.info(f"Attempting to download direct URL '{url}' to temp file: {local_path}")

            # Make the request, stream=True for potentially large files
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status() # Check for HTTP errors
                # Write content in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

        logging.info(f"✔️ Direct download successful: {local_path}")
        # Basic check
        if os.path.getsize(local_path) == 0:
            logging.info(f"⚠️ Warning: Direct download resulted in an empty file: {local_path}")
            # Optionally remove empty file?
            # os.remove(local_path)
            # return None
        return local_path

    except requests.exceptions.RequestException as e:
        logging.info(f"❌ Direct Download failed (Network/HTTP Error): {e}")
    except Exception as e:
        logging.info(f"❌ Direct Download failed (Other Error): {e}")

    # Cleanup on failure
    if local_path and os.path.exists(local_path):
        try: os.remove(local_path)
        except OSError: pass
    return None


# --- Helper Function: Process Video with TTS and Subtitles ---
def process_video_with_tts(base_video_url, audio_path, word_timings, topic, lang, copy_num, with_music=False ,platform='yt'):
    """
    Loads video (using downloaded path), adds TTS audio, loops/trims, adds subtitles.
    Returns the path to the final processed temporary video file, or None on failure.
    """
    # Initialize variables for cleanup
    final_video_clip = None
    temp_output_path = None
    base_video = None
    tts_audio = None
    back_music = None
    combined_audio = None
    looped_video = None
    processed_video = None
    resized_base_video = None
    clips_for_composite = []
    local_vid_path = None # To store the path of the downloaded base video

    try:
        # 1. Download the base video using yt-dlp
        # Note: Using the DIRECT URL fetched earlier by get_yt_dlp_info
        # It's passed as base_video_url here. Moviepy might handle some URLs directly,
        # but downloading first is often more reliable, especially for cloud functions.
        # Let's assume base_video_url *is* the direct URL here.
        # We need to download it.
        st.write(f"⏳ Downloading base video content from direct URL...")
        # Use the download_direct_url helper, or stick with yt-dlp if preferred
        # Using download_direct_url for potentially simpler direct downloads
        if platform == 'yt':
            local_vid_path = download_direct_url(base_video_url, suffix=".mp4")
            # Alternative: If yt-dlp download is preferred even for direct URLs:
            # local_vid_path = download_with_ytdlp(base_video_url, cookie_file_path=COOKIE_FILE_PATH)

            if not local_vid_path:
                raise ValueError(f"Failed to download base video content from: {base_video_url}")

            # 2. Load downloaded video with MoviePy
            st.write(f"➡️ Loading downloaded video: {local_vid_path}")
        elif platform in ('tk', 'pin'):
            # st.text('tk')
            # input()

            local_vid_path = download_with_ytdlp(base_video_url)
            if not local_vid_path:
                raise ValueError(f"Failed to download base video content from: {base_video_url}")
        # Ensure target_resolution is set for potential resizing during load
        with tempfile.NamedTemporaryFile(delete=False, suffix="_blurred.mp4") as tmp_blur_file:
            blurred_vid_path = tmp_blur_file.name
        if is_blur:
            try:
                st.text("blur_subtitles_in_video_unified")
                res = blur_subtitles_in_video_unified(
                    local_vid_path,
                    blurred_vid_path,
                    sample_time_sec=3.0,
                    ocr_min_confidence=10,
                    ocr_y_start_ratio=0.05, # Adjust if subtitles are higher/lower
                    ocr_padding=20,
                    blur_kernel_size=(51, 51), # Stronger blur
                    # tesseract_cmd_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    debug_save_frames=True # Set to True to see intermediate images
                )
                if res:
                    local_vid_path = blurred_vid_path
            except Exception as e:
                st.status(f"blur_subtitles_in_video_unified error: {e}")

        base_video = VideoFileClip(local_vid_path, audio=False, target_resolution=(720, 1280))

        video_duration = base_video.duration
        w = int(base_video.w) if base_video.w else 720
        h = int(base_video.h) if base_video.h else 1280
        st.write(f"✔️ Base video loaded: {w}x{h}, Duration: {video_duration:.2f}s")

        # 3. Load TTS Audio
        st.write(f"⏳ Loading TTS audio...")
        tts_audio = AudioFileClip(audio_path)
        audio_duration = tts_audio.duration
        if audio_duration <= 0:
             raise ValueError("TTS Audio has zero or negative duration.")

        # 4. Handle Background Music (Optional)
        combined_audio = tts_audio # Default
        if with_music:
            try:
                music_path_map ={ "sunrise" : "Sunrise.mp3",
                                  "ukulele" : "ukulele.mp3",
                                  "funk" : "funk.mp3"}


                

                # Ensure audio file exists relative to the script or provide full path
                music_path = os.path.join('audio', music_path_map[with_music]) # Assuming 'audio' subfolder
                if os.path.exists(music_path):
                    back_music = AudioFileClip(music_path).fx(afx.volumex, 0.08)
                    # Ensure music duration matches audio duration (loop or cut)
                    music_duration = back_music.duration
                    if music_duration < audio_duration:
                        num_loops = int(np.ceil(audio_duration / music_duration))
                        back_music = concatenate_audioclips([back_music] * num_loops).subclip(0, audio_duration)
                    elif music_duration > audio_duration:
                         back_music = back_music.subclip(0, audio_duration)

                    # Ensure both clips have the same duration before compositing
                    tts_audio = tts_audio.set_duration(audio_duration)
                    back_music = back_music.set_duration(audio_duration)

                    combined_audio = CompositeAudioClip([tts_audio, back_music])
                    st.write("✔️ Background music added.")
                else:
                    st.warning(f"Background music file not found at '{music_path}'. Skipping music.", icon="🎵")
            except Exception as music_err:
                st.warning(f"Could not load or process background music: {music_err}", icon="🎵")
        st.write(f"✔️ Audio loaded/prepared: Duration: {audio_duration:.2f}s")

        # 5. Resize Video Frame (Force 9:16)
        target_w, target_h = 720, 1280
        st.write(f"⏳ Resizing video to {target_w}x{target_h}...")
        try:
            # Use resize method for simplicity. Crop can be used for different framing.
            resized_base_video = base_video.resize(newsize=(target_w, target_h))
            st.write(f"✔️ Video resized.")
        except Exception as resize_err:
            st.warning(f"Could not resize video: {resize_err}. Using original.", icon="⚠️")
            resized_base_video = base_video # Fallback
            # Update target dimensions if fallback is used
            # target_w, target_h = w, h # This might break subtitle positioning/compositing size

        # 6. Loop or Trim Video to Match Audio Duration
        processed_video = resized_base_video # Start with the resized clip
        if video_duration < audio_duration:
            st.write(f"⏳ Looping video ({video_duration:.2f}s) to match audio ({audio_duration:.2f}s)...")
            num_loops = int(np.ceil(audio_duration / video_duration))
            # Create copies for concatenation
            clips_to_loop = [resized_base_video.copy().set_start(i * video_duration).set_duration(video_duration) for i in range(num_loops)]
            looped_video = concatenate_videoclips(clips_to_loop, method="compose")
            processed_video = looped_video.set_duration(audio_duration) # Explicitly set final duration
            st.write(f"✔️ Video looped {num_loops} times.")
        elif video_duration > audio_duration:
            st.write(f"⏳ Trimming video ({video_duration:.2f}s) to match audio ({audio_duration:.2f}s)...")
            processed_video = resized_base_video.subclip(0, audio_duration)
            st.write(f"✔️ Video trimmed.")
        else:
            st.write("✔️ Video duration matches audio duration.")

        # Ensure the processed video has the correct duration set
        processed_video = processed_video.set_duration(audio_duration)

        # 7. Set Audio
        final_video_clip = processed_video.set_audio(combined_audio)


        # 8. Generate and Add Subtitles
        st.write(f"⏳ Generating subtitles...")
        subtitle_clips_list = [] # Renamed to avoid conflict
        grouped_subs = group_words_with_timing(word_timings, words_per_group=SUBTITLE_WORDS_PER_GROUP)
        if grouped_subs:
            grouped_subs[-1]['end'] = audio_duration+1.5
            # ... (Subtitle generation loop using create_text_image, same as before) ...
            total_subs = len(grouped_subs)
            sub_progress_bar = st.progress(0)
            status_text = st.empty()

            for i, sub_data in enumerate(grouped_subs):
                text = sub_data['text']
                start = sub_data['start']
                end = sub_data['end']
                
                  
                sub_duration = end - start

                # if i == len(grouped_subs) +1 :
                #   sub_duration = 5

                
                  

                # Ensure start/end are within the final video duration
                if start >= audio_duration: continue # Skip subs starting after video ends
                if end > audio_duration: end = audio_duration # Trim subs ending after video ends
                sub_duration = end - start

                if not text.strip() or sub_duration <= 0.05: continue

                status_text.text(f"Creating subtitle {i+1}/{total_subs}: '{text[:30]}...'")
                text_img_array = create_text_image(
                     text.upper(), fontsize=SUBTITLE_FONT_SIZE, color=SUBTITLE_COLOR,
                     bg_color=SUBTITLE_BG_COLOR, font_path=SUBTITLE_FONT_PATH,
                     video_width=target_w # Use target width for wrapping
                )
                if text_img_array.shape[0] <= 10 or text_img_array.shape[1] <= 10:
                     st.warning(f"Skipping subtitle due to image creation error: '{text[:30]}...'")
                     continue

                subtitle_img_clip = ImageClip(text_img_array)\
                     .set_start(start)\
                     .set_duration(sub_duration)\
                     .set_position(('center', 'center')) # Center position

                subtitle_clips_list.append(subtitle_img_clip)
                sub_progress_bar.progress((i + 1) / total_subs)

            status_text.text(f"✔️ Generated {len(subtitle_clips_list)} subtitle clips.")

            # --- Composite final video with subtitles ---
            st.write("⏳ Compositing video and subtitles...")
            # Ensure the base clip is the first element
            clips_for_composite = [final_video_clip] + subtitle_clips_list
            # Explicitly set the size for CompositeVideoClip using the target dimensions
            final_video_clip = CompositeVideoClip(clips_for_composite, size=(target_w, target_h))
            st.write("✔️ Compositing complete.")
        else:
            st.warning("No valid word timings available to generate subtitles.", icon="⏱️")
            # final_video_clip remains the video with audio but no subtitles


        # 9. Export Final Video to a Temporary File
        st.write("⏳ Exporting final video...")

        # Create a temporary file path for the output video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"processed_output_") as temp_output_file_obj:
            temp_output_path = temp_output_file_obj.name

        # Ensure we have a valid clip object
        if not isinstance(final_video_clip, (VideoFileClip, CompositeVideoClip)):
             raise TypeError(f"Cannot write final video: Invalid clip object type {type(final_video_clip)}.")

        if final_video_clip.duration > audio_duration:
            st.write(f"⚠️ Clamping final clip duration from {final_video_clip.duration:.4f}s to audio duration {audio_duration:.4f}s to prevent glitches.")
            final_video_clip = final_video_clip.subclip(0, audio_duration)

        temp_audio_filename = f'temp-audio-{os.path.basename(temp_output_path)}.m4a'

        # Define ffmpeg parameters for web compatibility
        ffmpeg_params_list = [
            '-movflags', 'faststart', # Important for web streaming
            '-profile:v', 'high',     # H.264 profile
            '-level', '4.0',          # H.264 level
            # '-preset', 'medium',    # Already handled by MoviePy's preset argument
            # '-crf', '23',           # Optional: Constant Rate Factor for quality/size balance (lower is better quality, larger size)
        ]

        # Determine FPS, default to 24 if invalid
        output_fps = final_video_clip.fps if final_video_clip.fps and final_video_clip.fps > 0 else 24

        # Write the video file
        final_video_clip.write_videofile(
            temp_output_path,
            codec='libx264',          # Common, widely supported codec
            audio_codec='aac',        # Standard audio codec
            temp_audiofile=temp_audio_filename, # Use unique temp audio name
            remove_temp=True,         # Remove the temp audio file after processing
            fps=output_fps,
            preset='medium',          # Good balance of speed and quality
            threads=os.cpu_count() or 4, # Use available CPU cores
            logger='bar',             # Show progress bar in console/logs
            ffmpeg_params=ffmpeg_params_list
        )
        st.write(f"✔️ Final video exported to temporary path: {temp_output_path}")

        # Return the path to the successfully generated temporary file
        return temp_output_path

    except Exception as e:
        st.error(f"Error during video processing: {e}", icon="🎬")
        import traceback
        st.error(traceback.format_exc())
        # Ensure cleanup happens if an error occurred
        # Return None signifies failure
        return None # Explicitly return None on error

    finally:
        # --- Cleanup ---
        st.write("🧹 Cleaning up video processing resources...")
        # Use try-except for each close/remove operation for robustness
        resources_to_close = [
             base_video, tts_audio, back_music, combined_audio, looped_video,
             processed_video, resized_base_video, final_video_clip
        ] + subtitle_clips_list # Add subtitle clips to the list

        for resource in resources_to_close:
             if resource:
                  try: resource.close()
                  except Exception as close_err: pass # Ignore errors during cleanup

        # Delete the downloaded base video temp file
        if local_vid_path and os.path.exists(local_vid_path):
             try:
                  os.remove(local_vid_path)
                  # st.write(f"🧹 Deleted temp base video: {local_vid_path}") # Verbose log
             except Exception as rm_err: st.warning(f"Could not delete temp base video: {local_vid_path} ({rm_err})")

        # Delete the main output temp file only if an error occurred before returning path
        # If function succeeded, the caller is responsible for the returned path.
        # If it failed, temp_output_path might exist but is useless.
        # Let's check if the return value would have been None.
        # This is tricky. Best practice: Caller cleans up the returned path.
        # Here, just ensure MoviePy's internal temp files are cleaned if possible.

        # Attempt to remove the explicitly named temp audio file if it exists
        if 'temp_audio_filename' in locals() and os.path.exists(temp_audio_filename):
             try:
                  os.remove(temp_audio_filename)
                  # st.write(f"🧹 Deleted temp audio file: {temp_audio_filename}") # Verbose log
             except Exception as rm_audio_err:
                  st.warning(f"Could not remove temp audio file {temp_audio_filename}: {rm_audio_err}")

        st.write("🧹 Cleanup finished.")


# --- Helper Function: Upload Video to S3 ---
def upload_pil_image_to_s3(
    image,
    bucket_name,
    aws_access_key_id,
    aws_secret_access_key,
    object_name="",
    region_name=None,
    endpoint_url=None,
    image_format="JPEG",
):
    """Upload a PIL image to any S3-compatible bucket (Backblaze B2)."""
    clean_region = region_name.strip() if region_name else None
    clean_endpoint = endpoint_url.strip() if endpoint_url else None
    bucket = bucket_name.strip() if bucket_name else ""
    if not bucket:
        st.error("Bucket name is required for image upload.", icon="🪣")
        return None
    if not aws_access_key_id or not aws_secret_access_key:
        st.error("AWS credentials not available for image upload.", icon="🔑")
        return None

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id.strip(),
            aws_secret_access_key=aws_secret_access_key.strip(),
            region_name=clean_region,
            endpoint_url=clean_endpoint,
        )
    except Exception as e:
        st.error(f"Failed to initialize S3 client for image upload: {e}", icon="🚨")
        return None

    if not object_name:
        object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"
    object_key = object_name.lstrip("/")

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image_format.upper(), quality=90)
    img_byte_arr.seek(0)

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=img_byte_arr,
            ContentType=f"image/{image_format.lower()}",
        )
    except Exception as e:
        st.error(f"S3 image upload failed: {e}", icon="🖼️")
        return None

    safe_object_name = urllib.parse.quote(object_key, safe="/")
    if clean_endpoint:
        return f"{clean_endpoint.rstrip('/')}/{bucket}/{safe_object_name}"
    region_segment = clean_region or "us-east-1"
    if region_segment == 'us-east-1':
        return f"https://{bucket}.s3.amazonaws.com/{safe_object_name}"
    return f"https://{bucket}.s3.{region_segment}.amazonaws.com/{safe_object_name}"


def upload_vid_to_s3(s3_cli, video_path, bucket_name, object_name, region_name, endpoint_url=None):
    """Uploads a video file to S3."""
    if not s3_cli:
        st.error("S3 Client not initialized. Cannot upload.", icon="🚫")
        return None
    if not video_path or not os.path.exists(video_path): # Check if path is valid and exists
        st.error(f"Video file not found or path invalid: {video_path}", icon="❌")
        return None
    if os.path.getsize(video_path) == 0:
        st.error(f"Video file is empty, cannot upload: {video_path}", icon="⚠️")
        return None

    clean_region = region_name.strip() if isinstance(region_name, str) else region_name
    clean_endpoint = endpoint_url.strip() if endpoint_url else None
    bucket = bucket_name.strip() if isinstance(bucket_name, str) else bucket_name
    object_key = object_name.lstrip("/") if isinstance(object_name, str) else object_name

    st.write(f"☁️ Uploading '{object_name}' to S3 bucket '{bucket_name}'...")
    try:
        with open(video_path, "rb") as video_file:
            s3_cli.put_object(
                Bucket=bucket,
                Key=object_key,
                Body=video_file,
                ContentType='video/mp4' # Ensure correct MIME type
            )

        # Construct the S3 URL (ensure object_name is URL-encoded)
        safe_object_name = urllib.parse.quote(object_key, safe="/")
        if clean_endpoint:
            video_url = f"{clean_endpoint.rstrip('/')}/{bucket}/{safe_object_name}"
        else:
            region_segment = clean_region or 'us-east-1'
            if region_segment == 'us-east-1':
                 # us-east-1 URL format might differ (no region needed or optional)
                 # Check S3 documentation or common practice. Using the format without region is often safe.
                 video_url = f"https://{bucket}.s3.amazonaws.com/{safe_object_name}"
                 # Alternate: f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{safe_object_name}"
            else:
                 video_url = f"https://{bucket}.s3.{region_segment}.amazonaws.com/{safe_object_name}"

        st.success(f"✔️ Video uploaded to S3: {object_name}")
        st.write(f"🔗 S3 URL: {video_url}")
        return video_url
    except NoCredentialsError:
        st.error("AWS Credentials not available for S3 upload.", icon="🔒")
        return None
    except Exception as e:
        st.error(f"S3 Upload Error: {e}", icon="☁️")
        import traceback
        st.error(traceback.format_exc()) # Log full traceback for debugging
        return None

# --- Helper: Sync Data Editor State ---
# Define sync_search_data function (keep the improved version from previous response)
def sync_search_data():
    """ Safely syncs the data_editor state back to session_state.search_data """
    try:
        editor_state = st.session_state.get("search_topic_editor", {})
        edited_rows = editor_state.get("edited_rows", {})
        added_rows = editor_state.get("added_rows", [])
        deleted_rows = editor_state.get("deleted_rows", [])

        # Use the snapshot from *before* the editor was rendered
        if 'search_data_snapshot' in st.session_state:
            current_df = st.session_state.search_data_snapshot.copy()
        else:
            # Fallback: use current state, but this might contain unsynced changes
            current_df = st.session_state.search_data.copy() if isinstance(st.session_state.search_data, pd.DataFrame) else pd.DataFrame([{'Topic': '', 'Language': 'English','Search Term': '',  "Script Angle": "default_clean", 'Video Results': 5, 'BG Music' : "sunrise", 'TTS Voice': 'sage'}])

        # Apply deletions first (indices are based on the snapshot)
        valid_delete_indices = sorted([idx for idx in deleted_rows if idx < len(current_df)], reverse=True)
        if valid_delete_indices:
            current_df = current_df.drop(index=valid_delete_indices).reset_index(drop=True)

        # Apply edits
        for idx_str, changes in edited_rows.items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(current_df): # Check index validity against potentially shrunk df
                    for col, val in changes.items():
                        if col in current_df.columns:
                            # Handle potential type issues from editor
                            if col == 'Video Results':
                                 try: current_df.loc[idx, col] = int(val) if pd.notna(val) else 5
                                 except (ValueError, TypeError): current_df.loc[idx, col] = 5 # Fallback
                            else:
                                 current_df.loc[idx, col] = str(val) if pd.notna(val) else ""

            except ValueError: pass # Ignore invalid index strings

        # Apply additions
        if added_rows:
            expected_cols = ["Topic", "Language", "Search Term", "Script Angle", "Video Results","BG Music","TTS Voice"]
            processed_adds = []
            for row_dict in added_rows:
                 if isinstance(row_dict, dict):
                      new_row = {}
                      for col in expected_cols:
                           val = row_dict.get(col)
                           # Set defaults or convert type
                           if col == "Language":
                                new_row[col] = str(val).strip() if pd.notna(val) and str(val).strip() else "English"
                           elif col == "Script Angle":
                                new_row[col] = str(val).strip() if pd.notna(val) and str(val).strip() else "default"
                           elif col == "Video Results":
                                try: new_row[col] = int(val) if pd.notna(val) else 5
                                except (ValueError, TypeError): new_row[col] = 5
                           elif col == "BG Music":
                                new_row[col] = val if pd.notna(val) and val != '' else False
                           elif col == "TTS Voice":
                                new_row[col] = str(val).strip() if pd.notna(val) and str(val).strip() else "sage"
                           else:
                                new_row[col] = str(val).strip() if pd.notna(val) else ""
                      processed_adds.append(new_row)
            if processed_adds:
                 add_df = pd.DataFrame(processed_adds)
                 current_df = pd.concat([current_df, add_df], ignore_index=True)


        # Final cleanup and validation of the entire DataFrame
        expected_cols = ["Topic",  "Language", "Search Term","Script Angle", "Video Results","BG Music","TTS Voice"]
        for col in expected_cols: # Ensure all columns exist
             if col not in current_df.columns:
                  if col == "Language": current_df[col] = "English"
                  elif col == "Script Angle": current_df[col] = "default_clean"
                  elif col == "Video Results": current_df[col] = 5
                  else: current_df[col] = ""
        current_df = current_df[expected_cols] # Enforce column order

        # Type conversion and validation
        try:
            current_df['Video Results'] = pd.to_numeric(current_df['Video Results'], errors='coerce').fillna(5).astype(int)
            current_df['Video Results'] = current_df['Video Results'].apply(lambda x: max(0, min(x, 100))) # Clamp 1-50
        except Exception: current_df['Video Results'] = 5
        current_df['Topic'] = current_df['Topic'].fillna('').astype(str).str.strip()
        current_df['Search Term'] = current_df['Search Term'].fillna('').astype(str).str.strip()
        current_df['Language'] = current_df['Language'].fillna('English').astype(str).str.strip()
        current_df['Script Angle'] = current_df['Script Angle'].fillna('default_clean').astype(str).str.strip()

        # Ensure 'Script Angle' values are valid options
        # current_df['Script Angle'] = current_df['Script Angle'].apply(lambda x: x if x in SCRIPT_VER_OPTIONS else 'default')


        # Remove rows where both Topic and Search Term are empty, unless it's the only row left
        meaningful_rows_filter = (current_df['Topic'] != '') | (current_df['Search Term'] != '')
        if len(current_df) > 1:
            current_df = current_df[meaningful_rows_filter]

        # If empty after filtering, add back a default row
        if current_df.empty:
             current_df = pd.DataFrame([{'Topic': '',  'Language': 'English','Search Term': '', "Script Angle": "default_clean", 'Video Results': 5, 'BG Music' : "sunrise", 'TTS Voice': 'sage'}])

        # Update the main session state and create a fresh snapshot
        st.session_state.search_data = current_df.reset_index(drop=True)
        st.session_state.search_data_snapshot = st.session_state.search_data.copy()
        # logging.info("Data editor sync complete.") # Debug log
    except Exception as sync_err:
         st.error(f"Error syncing data editor state: {sync_err}")
         # Fallback: Keep the previous state? Or reset?
         if 'search_data_snapshot' in st.session_state:
              st.session_state.search_data = st.session_state.search_data_snapshot.copy()
         else: # If snapshot missing too, hard reset
              st.session_state.search_data = pd.DataFrame([{'Topic': '',  'Language': 'English','Search Term': '', "Script Angle": "default_clean", 'Video Results': 5, 'BG Music' : False, 'TTS Voice': 'sage'}])
              st.session_state.search_data_snapshot = st.session_state.search_data.copy()


# --- Helper: Convert DF to CSV ---
@st.cache_data # Cache the conversion result for a given DataFrame
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to CSV bytes."""
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    return output.getvalue()

# <<< END: IMPORT STATEMENTS AND HELPER FUNCTIONS >>>


# --- Streamlit App ---

st.title("🎬 YouTube Video Selector & TTS Generator")
st.caption("Search YouTube, select videos (multiple times allowed), generate TTS script, create subtitled videos, and upload to S3.")

# --- Session State Initialization ---
# Stores data for each *selected video job instance* {job_key: video_data_dict}
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {}
# Stores results from the last API search {search_key: result_data}
if 'api_search_results' not in st.session_state:
    st.session_state.api_search_results = {}
# Input DataFrame for search terms and topics
if 'search_data' not in st.session_state:
    st.session_state.search_data = pd.DataFrame([
        {'Topic': 'sofa sale', 'Language': 'English','Search Term': 'auto',  "Script Angle": "default_clean", 'Video Results': 40, 'BG Music' : "sunrise", 'TTS Voice': 'sage'}
    ])
# Snapshot for data editor comparison
if 'search_data_snapshot' not in st.session_state:
    st.session_state.search_data_snapshot = st.session_state.search_data.copy()
# Flag to indicate if a search has been run
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

if 'search_more_manual_input_visible' not in st.session_state:
    st.session_state.search_more_manual_input_visible = {} # To store visibility per search_key
if 'search_more_manual_query' not in st.session_state:
    st.session_state.search_more_manual_query = {} # To store the manual query text per search_key

# --- State for Batch Processing ---
if 'generation_queue' not in st.session_state: st.session_state.generation_queue = []
if 'batch_processing_active' not in st.session_state: st.session_state.batch_processing_active = False
if 'batch_total_count' not in st.session_state: st.session_state.batch_total_count = 0
if 'batch_processed_count' not in st.session_state: st.session_state.batch_processed_count = 0
if 'resolved_vid_urls' not in st.session_state: st.session_state.resolved_vid_urls = {}


# --- Input Area (Sidebar) ---
st.sidebar.header("Inputs & Actions")
st.sidebar.write("Enter Search Terms and Topics:")

# Take snapshot *before* rendering the editor
st.session_state.search_data_snapshot = st.session_state.search_data.copy()

edited_df = st.sidebar.data_editor(
    st.session_state.search_data, # Use the main state data
    column_config={
        "Script Angle": st.column_config.SelectboxColumn("Script Angle", options=SCRIPT_VER_OPTIONS, default="default_clean", required=True),
        "Video Results": st.column_config.NumberColumn("Video Results", min_value=0, max_value=100, step=1, default=5, required=True),
        "Language": st.column_config.TextColumn("Language", default="English", required=True),
        "Topic": st.column_config.TextColumn("Topic"),
        "Search Term": st.column_config.TextColumn("Search Term (or 'auto')"),
        "BG Music": st.column_config.SelectboxColumn("BG Music", options=BG_VER_OPTIONS, default=False, required=True),
        "TTS Voice": st.column_config.SelectboxColumn("TTS Voice", options=TTS_VOICE_OPTIONS, default="sage", required=True)
        },
    num_rows="dynamic",
    use_container_width=True,
    key="search_topic_editor",
    # on_change=sync_search_data # Sync *after* edits are registered by Streamlit
)

# --- Action Buttons (Sidebar) ---
col1, col2 = st.sidebar.columns(2)
# Use on_click for search button to ensure sync happens *before* search logic runs
search_button = col1.button(
    "🔍 Search Videos",
    use_container_width=True,
    disabled=st.session_state.batch_processing_active,
    on_click=sync_search_data # Sync now happens via on_change of editor
)
clear_button = col2.button("🧹 Clear All", use_container_width=True, type="secondary", disabled=st.session_state.batch_processing_active)
# with_music = col1.checkbox("With BG music?", value=False)
# with_music_rand = col2.checkbox("With BG music randomly?", value=False)
is_youtube = col1.checkbox("YT") 
is_tiktok = col1.checkbox("tk")
is_pinterest = col1.checkbox("Pin")
is_blur = col2.checkbox("Blur captions?")
if clear_button:
    # Reset all relevant states
    st.session_state.selected_videos = {}
    st.session_state.search_triggered = False
    st.session_state.api_search_results = {}
    st.session_state.generation_queue = []
    st.session_state.batch_processing_active = False
    st.session_state.batch_total_count = 0
    st.session_state.batch_processed_count = 0
    st.session_state.resolved_vid_urls = {}
    st.session_state.search_data = pd.DataFrame([{'Topic': '','Language' : 'English', 'Search Term': '',"Script Angle" : "default_clean", 'Video Results': 5}])
    st.session_state.search_data_snapshot = st.session_state.search_data.copy()
    # Clear the editor state explicitly if possible (might require widget key manipulation - complex)
    # For now, resetting search_data and rerunning should clear the editor visually.
    st.success("Selections, results, and generation state cleared!", icon="✅")
    st.rerun()

# --- Global Process Button ---
st.sidebar.divider()
ready_jobs_count = sum(
    1 for job_key, data in st.session_state.selected_videos.items()
    if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
)
process_all_button = st.sidebar.button(
    f"🎬 Process {ready_jobs_count} Ready Job{'s' if ready_jobs_count != 1 else ''}",
    use_container_width=True,
    type="primary",
    disabled=ready_jobs_count == 0 or st.session_state.batch_processing_active
)

if process_all_button:
    st.session_state.sheet_appended = False


  
    # Find job keys ready for processing
    job_keys_to_process = [
        job_key for job_key, data in st.session_state.selected_videos.items()
        if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
    ]
    if job_keys_to_process:
        st.session_state.generation_queue = job_keys_to_process
        st.session_state.batch_processing_active = True
        st.session_state.batch_total_count = len(job_keys_to_process)
        st.session_state.batch_processed_count = 0
        st.sidebar.info(f"Queued {len(job_keys_to_process)} generation jobs.", icon="⏳")
        # Update status and clear errors for queued jobs
        for job_key in job_keys_to_process:
            if job_key in st.session_state.selected_videos:
                st.session_state.selected_videos[job_key]['Generation Error'] = None
                st.session_state.selected_videos[job_key]['Generated S3 URL'] = None
                st.session_state.selected_videos[job_key]['Status'] = 'Queued'
        st.rerun()
    else:
        st.sidebar.warning("No selected video jobs are ready for processing (need Direct URL).", icon="⚠️")

if st.sidebar.button("Close all previews ✖",use_container_width=True,disabled= st.session_state.batch_processing_active):
  show_player_list=[]
  for item in st.session_state:
    if "show_player_" in item and st.session_state[item] == True:
      st.session_state[item] = False


    


# st.text(show_player_list)

st.sidebar.info("Use '➕ Select' to queue a generation job. Each click adds one job.", icon="ℹ️")
# st.sidebar.info(f"{type(st.session_state)}", icon="ℹ️")
# st.sidebar.warning("Video generation can take several minutes per job.", icon="⏱️")

# --- Processing Logic ---

# 1. Handle Search Button Click (Logic moved slightly)
if search_button: # This block runs when the button is clicked
    # sync_search_data() # Sync is now handled by on_change of editor

    # Perform validation on the synced data
    search_df = st.session_state.search_data # Use the already synced data
    valid_input = True
    if search_df.empty:
        st.sidebar.warning("Input table is empty.", icon="⚠️")
        valid_input = False
    # Check if required columns have at least one non-empty value after sync/validation in sync_search_data
    elif ((search_df['Topic'] == '') & (search_df['Search Term'] == '')).all():
         if len(search_df) > 1: # Allow single empty row, but not multiple
            st.sidebar.warning("Please provide a 'Topic' or 'Search Term' in at least one row.", icon="⚠️")
            valid_input = False
         # Else: allow the single default empty row.
    # Check 'auto' term requires Topic?
    auto_rows = search_df[search_df['Search Term'].str.lower() == 'auto']
    if not auto_rows.empty and (auto_rows['Topic'] == '').any():
        st.sidebar.warning("Rows with 'auto' in 'Search Term' must have a non-empty 'Topic'.", icon="⚠️")
        valid_input = False

    if valid_input:
        st.sidebar.success("Input valid, proceeding with search.")
        st.session_state.search_triggered = True
        st.session_state.api_search_results = {} # Clear previous results
        st.session_state.current_search_df = search_df.copy() # Store validated DF for search step
        st.rerun() # Rerun to trigger the API search section
    # No else needed, warnings displayed above

# 2. Perform API Search if Triggered
if st.session_state.search_triggered and 'current_search_df' in st.session_state:
    search_df = st.session_state.current_search_df
    search_items = search_df.to_dict('records')

    st.info(f"Searching API for {len(search_items)} topic/term row(s)...", icon="🔍")
    progress_bar = st.progress(0)
    status_text_api = st.empty()

    api_error_occurred = False
    results_cache = {} # Temp cache for *this* search run

    platform_options_base = []
    if is_youtube:
        platform_options_base.append(('yt', "YouTube"))
    if is_tiktok:
        platform_options_base.append(('tk', "TikTok"))
    if is_pinterest:
        platform_options_base.append(('pin', "Pinterest"))

    if not platform_options_base:
        st.warning("Please select at least one platform before searching.", icon="⚠️")
        st.session_state.search_triggered = False
    else:
        for i, item in enumerate(search_items):
            # Retrieve validated data from the DataFrame row
            term = item['Search Term'] # Already stripped in sync
            topic = item['Topic']     # Already stripped in sync
            count = item['Video Results'] # Already int and clamped
            lang = item['Language']   # Already stripped in sync
            script_ver = item["Script Angle"] # Already stripped in sync
            bg_music = item["BG Music"]
            tts_voice = item["TTS Voice"]
            og_term = term

            if count <= 0:
                st.info(f"Skipping row {i+1} ('{topic}' / '{term}') because Video Results is set to 0.", icon="ℹ️")
                continue
            # Handle 'auto' search term generation
            if term.lower() == 'auto':
                if not topic: # Should be caught by earlier validation, but double-check
                    st.warning(f"Skipping row {i+1}: 'auto' search term requires a Topic.", icon="⚠️")
                    continue
                status_text_api.text(f"Generating search terms for: '{topic}'...")
                try:
                    # --- Use the refined GPT prompt for search terms ---
                    # generated_term = chatGPT(f"""You are a viral video ad expert. I will give you a topic, and you will return the top 3 YouTube Shorts search terms that:
    
                    #   - Are short (2–5 words)
    
                    #   - Clearly describe what viewers will see in the video (visuals only)
    
                    #   - Lead to emotionally engaging, surprising, or curiosity-triggering content
    
                    #   - Are perfect for remixing or using as inspiration for Facebook video ads
    
                    #   - Focus on things like transformations, objects in motion, satisfying actions, luxury aesthetics, clever space-saving, or unexpected reveals
    
                    #   - Avoid abstract or advice-based phrases (like “tips,” “hacks,” or “secrets”)
    
                    #   - Avoid using non visual\describing words that are not likely to be relevent (like 'On credit', "Financing", etc)
    
                    #   - Add '#shorts' to the end of each search term and separate terms with ' | '
    
                    #   - if the topic is a service (like lawyer) that is intangible, think of something else that can be used (like "Veterans Benefits Lawyer free consultation" give "veteran shares #shorts ")
    
                    #   Example:
    
                    #   Input: sofa
    
                    #   Output:
    
                    #   'sofa transformation #shorts | hidden bed sofa #shorts | luxury sofa unboxing #shorts'
    
                    #   Input: car finance bad credit no deposit
                    #   Output:
    
                    #   'new car tour #shorts | car reaction #shorts | new car surprise #shorts'
    
                    #   My topic:
    
                    # {topic}""", client=openai_client, model="gpt-4") # Use your full validated prompt  
    
                    # generated_term = gemini_text_lib(f"""
                    #                 You are a Viral Video Ad Scout. Your mission is to find YouTube Shorts search terms that uncover visually compelling, user-generated style content perfect for remixing into high-performing Facebook video ads. The key is to think about what *actual users* are likely to upload as Shorts – authentic, engaging moments rather than polished ads.
    
                    #                 Given a topic, return the top 4 YouTube Shorts search terms that meet these criteria 1 of them is the topic itself as consice as possible, the others:
    
                    #                 1.  **Concise & Visual:** 2-3 words, clearly describing *tangible actions, objects, or visual transformations* viewers will see. Focus on the visual verb or noun.
                    #                 2.  **Emotionally Resonant:** Leads to content triggering surprise, satisfaction, curiosity, awe, or joy. Think "wow moments."
                    #                 3.  **Remix-Ready:** Content should be inspiring for new ad creatives, focusing on:
                    #                     * **Transformations:** Before & after, makeovers, redesigns.
                    #                     * **Objects in Motion/Use:** Product demos (organic feel), gadgets in action, vehicles moving.
                    #                     * **Satisfying Processes:** Cleaning, organizing, creating, ASMR-like actions.
                    #                     * **Luxury & Aesthetics:** Unboxings, showcases of high-end items, beautiful setups.
                    #                     * **Clever Solutions:** Space-saving ideas, innovative uses, smart designs.
                    #                     * **Unexpected Reveals:** Hidden features, surprise elements, sudden changes.
                    #                 4.  **Authentic YouTube Style:** Prioritize terms that reflect genuine user uploads, not overly commercial or "how-to" content.
                    #                 5.  **Avoid:**
                    #                     * Abstract concepts, advice-based phrases (e.g., "tips," "hacks," "secrets," "how to learn").
                    #                     * Non-visual qualifiers or descriptive words unlikely to be in a visual search (e.g., "on credit," "financing," "affordable," "best"). The visual should speak for itself.
                    #                 6.  **Handling Intangible Services/Topics:**
                    #                     * If the topic is a service (e.g., lawyer, insurance, software), focus on *visual proxies or relatable human experiences/outcomes* associated with it.
                    #                     * Example: For "Veterans Benefits Lawyer," think about the *result* or *emotion*. Instead of "lawyer consultation," terms like: "veteran disability approved #shorts" or "soldier homecoming surprise #shorts". For software, "dashboard animation #shorts" or "app feature showcase #shorts".
                    #                 7.  **Format:**
                    #                     * Add '#shorts' to the end of each search term.
                    #                     * Separate terms with ' | '.
    
                    #                 Example 1:
                    #                 Input: sofa
                    #                 Output: 'sofa transformation #shorts | hidden storage sofa #shorts | modular sofa setup #shorts| sofa #shorts'
    
                    #                 Example 2:
                    #                 Input: car finance bad credit no deposit
                    #                 Output: 'new car day reaction #shorts | dream car surprise #shorts | first car celebration #shorts | car #shorts'
    
                    #                 Example 3:
                    #                 Input: home cleaning service
                    #                 Output: 'dirty to clean house #shorts | satisfying home clean #shorts | messy room makeover #shorts | home cleaning service #shorts'
                    #                 return just the output no intros or explaining
                    #                 My topic: {topic}
                    #                 """,
    
                    #                 model = "gemini-2.5-flash-preview-04-17"
                    #                 # client=openai_client
                    #                 ) 
                    generated_term= chatGPT(f"""{AUTO_TERMS_PROMPT} {topic}""",model="gpt-5",client=openai_client) #, model ='gemini-2.5-flash-preview-04-17'
                    if not generated_term:
                        st.warning(f"Failed to generate search terms for '{topic}'. Skipping.", icon="🤖")
                        continue
                    term = generated_term # Replace 'auto' with generated terms
                    st.write(f"Generated terms for '{topic}': {term}")
                except Exception as gpt_err:
                    st.error(f"Error generating search terms for '{topic}': {gpt_err}", icon="🤖")
                    continue
    
            platform_options = platform_options_base
            platform_used = False
            for platform_code, platform_label in platform_options:
                platform_used = True
                unique_search_key = f"{term}_{topic}_{lang}_{script_ver}_{platform_code}"
                if unique_search_key in results_cache:
                    continue
    
                status_text_api.text(f"Searching {platform_label} for: '{term}' (Topic: '{topic}')...")
                videos = None
                next_block_start_index = None
    
                if platform_code == 'yt':
                    videos = search_youtube(youtube_api_key_secret, term, count)
                elif platform_code == 'tk':
                    videos_list, next_block_start_index = search_tiktok_links_google(
                        api_keys=youtube_api_key_secret,
                        cx_id=GOOGLE_TIKTOK_CX_ID,
                        query=term,
                        num_results=count,
                        start_index=1
                    )
                    videos = videos_list
                elif platform_code == 'pin':
                    videos_list, next_block_start_index = search_pinterest_links_google(
                        api_keys=youtube_api_key_secret,
                        cx_id=GOOGLE_PINTEREST_CX_ID,
                        query=term,
                        num_results=count,
                        start_index=1,
                        # videos_only=pin_videos_only
                    )
                    videos = videos_list
    
                if videos is None:
                     st.error(f"Stopping search due to critical API issue for {platform_label}.", icon="🚫")
                     api_error_occurred = True
                     break
                if not videos:
                     st.warning(f"No {platform_label} results for '{term}' (Topic: '{topic}'). Skipping this platform.", icon="⚠️")
                     continue

                if platform_code == 'yt':
                    results_cache[unique_search_key] = {
                        'videos': videos,
                        'topic': topic,
                        'lang': lang,
                        "script_ver": script_ver,
                        'original_term': term,
                        'bg_music': bg_music,
                        'original_input_count': count,
                        'tts_voice': tts_voice,
                        'input_search_term': og_term,
                        'platform': 'yt'
                    }
                elif platform_code == 'tk':
                    results_cache[unique_search_key] = {
                        'videos': videos,
                        'topic': topic,
                        'lang': lang,
                        "script_ver": script_ver,
                        'original_term': term,
                        'bg_music': bg_music,
                        'original_input_count': count,
                        'tts_voice': tts_voice,
                        'input_search_term': og_term,
                        'platform': 'tk',
                        'next_start_index': next_block_start_index,
                        'last_start_index': 1
                    }
                elif platform_code == 'pin':
                    results_cache[unique_search_key] = {
                        'videos': videos,
                        'topic': topic,
                        'lang': lang,
                        "script_ver": script_ver,
                        'original_term': term,
                        'bg_music': bg_music,
                        'original_input_count': count,
                        'tts_voice': tts_voice,
                        'input_search_term': og_term,
                        'platform': 'pin',
                        'next_start_index': next_block_start_index,
                        'last_start_index': 1
                    }
                
                time.sleep(0.1) # Brief pause
    
            if not platform_used:
                st.warning("Please select at least one platform before searching.", icon="⚠️")
                api_error_occurred = True
                break
    
            if api_error_occurred:
                break
    
        progress_bar.progress((i + 1) / len(search_items))

    status_text_api.text("API Search complete." if not api_error_occurred else "API Search halted due to error.")
    st.session_state.api_search_results = results_cache # Update main cache
    st.session_state.search_triggered = False # Reset trigger
    # No st.rerun() here, let results display immediately


# 3. Display Search Results
st.divider() # Separator before results
if st.session_state.api_search_results:
    st.subheader("Search Results & Video Selection (Grid View)")

    # The search_key should uniquely identify an input configuration from your data editor
    for search_key, result_data in st.session_state.api_search_results.items():
        videos = result_data.get('videos', []) # List of video dicts
        topic_for_group = result_data['topic']
        lang_for_group = result_data['lang']
        script_ver_for_group = result_data["script_ver"]
        # This is the term (or pipe-separated terms) that were *actually* used for the most recent search for this group.
        # It will be updated if "Search More" is used.
        current_search_terms_for_group_display = result_data['original_term']
        bg_music_for_group = result_data.get('bg_music', False) # Default if not present
        tts_voice_for_group = result_data.get('tts_voice', 'sage') # Default if not present
        platform_for_group = result_data.get('platform', 'yt')
        # These are from the original data editor input for this topic/row
        # Ensure these keys ('input_search_term', 'original_input_count') were stored when api_search_results was populated
        input_search_term_from_editor = result_data.get('input_search_term', current_search_terms_for_group_display)
        count_from_editor = result_data.get('original_input_count', MAX_RESULTS_PER_QUERY) # Fallback to global max

        # Initialize state for this specific search group's "Search More" manual input, if not already
        st.session_state.search_more_manual_input_visible.setdefault(search_key, False)
        st.session_state.search_more_manual_query.setdefault(search_key, "")

        term_container = st.container(border=True)
        with term_container:
            st.subheader(f"Results for Search: \"{current_search_terms_for_group_display}\"")
            st.caption(f"(Original Topic: \"{topic_for_group}\", Lang: {lang_for_group}, Angle: {script_ver_for_group}, Input Term: '{input_search_term_from_editor}')")

            if not videos:
                st.write("No videos found for this search.")
            else:
                num_videos = len(videos)
                num_cols = 3 # Adjust number of columns as desired
                if platform_for_group in ('tk', 'pin'):
                    num_cols = 5

                for i in range(0, num_videos, num_cols):
                    cols = st.columns(num_cols, gap="small", vertical_alignment="top")
                    for j in range(num_cols):
                        video_index = i + j
                        if video_index < num_videos:
                            video = videos[video_index]
                            with cols[j]:
                                # st.text(video)
                                video_id = video['videoId']
                                video_title = video['title']
                                thumbnail = video.get('thumbnail_url',"")
                                standard_video_url = video.get('url') # Standard YT URL
                                grid_instance_key = f"vid_{video_id}_{search_key}_{i}_{j}" # More specific key
                                platform = video['platform']
                                show_video_key = f"show_player_{grid_instance_key}"
                                st.session_state.setdefault(show_video_key, False)

                                st.write(f"**{textwrap.shorten(video_title, width=30, placeholder='...')}**")
                                # st.caption(f"ID: {video_id}")

                                if st.session_state[show_video_key]:
                                    try:
                                        if platform == 'yt':
                                            iframe_code = f"""
                                            <iframe width="350" height="560"
                                            src="https://www.youtube.com/embed/{video_id}"
                                            title="YouTube video player" frameborder="0"
                                            allow="accelerometer; autoplay; clipboard-write; encrypted-media;
                                            gyroscope; picture-in-picture; web-share"
                                            allowfullscreen></iframe>"""
                                        elif platform == 'tk':
                                                original_width = 300
                                                original_height = 560
                                                target_width = 210
                                                scale = target_width / original_width
                                                scaled_height = int(original_height * scale)  
                                                # src="https://www.tiktok.com/embed/v2/{video_id}?lang=en-US&referrer=https%3A%2F%2Fapp.onecompiler.com%2F43xy3g8ex_43y3d4ztj%2F" 
                                                # src="https://www.tiktok.com/embed/v2/{video_id}?flag=creativeCenterConfig&token=567a7e0b41293818d525d57a7f552e8f" 
                                                iframe_code = f"""
                                                <!-- Visible wrapper: clips excess space -->
                                                <div style="width: {target_width}px; height: {scaled_height}px; overflow: hidden;">

                                                <!-- Inner container: real TikTok player size scaled down -->
                                                <div style="width: {original_width}px; height: {original_height}px; transform: scale({scale}); transform-origin: top left;">
                                                    <iframe 
                                                    src="https://www.tiktok.com/embed/v3/{video_id}?lang=en-US&referrer=https%3A%2F%2Fapp.onecompiler.com%2F43xy3g8ex_43y3d4ztj%2F&autoplay=1" 
                                                    allow="autoplay"
                                                    width="100%" 
                                                    height="100%" 
                                                    allowfullscreen 
                                                    scrolling="no" 
                                                    loading="lazy"
                                                    style="border:none; border-radius: 12px;">
                                                    </iframe>
                                                </div>

                                                </div>

                                                """
                                        elif platform == 'pin':
                                                target_width = 236
                                                target_height = 420
                                                video_id_markdown =  video_id.split("-")[-1]
                                                iframe_code = f"""
                                                <div style="width:{target_width}px; margin:auto;">
                                                    <iframe 
                                                        src="https://assets.pinterest.com/ext/embed.html?id={video_id_markdown}"
                                                        height="{target_height}"
                                                        width="{target_width}"
                                                        frameborder="0"
                                                        scrolling="no"
                                                        loading="lazy"
                                                        style="border-radius:12px;">
                                                    </iframe>
                                                </div>
                                                """
#                                                 iframe_code = f"""
#                                             <iframe 
#                                             height="400" 
#                                             loading="lazy"
#                                             width="220" 
#                                             src="https://www.tiktok.com/embed/v3/{video_id}?autoplay=1"
#                                             allowfullscreen 
#                                             allow="encrypted-media"
#                                             scrolling="no"
#                                             style="overflow: hidden; border: none;"
#                                             title="TikTok Embed"
#                                             ></iframe>




                                            
# """
     

                                        st.markdown(iframe_code, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Video preview failed: {e}")
                                else:
                                    if platform == 'yt':
                                        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                                    elif platform in ('tk', 'pin'):
                                        thumbnail_url = thumbnail or standard_video_url

                                    try:
                                        if platform == 'yt':
                                            st.image(thumbnail_url, use_container_width=False, caption="Video Thumbnail",width=200)
                                        elif platform in ('tk', 'pin'):
                                          # st.image(thumbnail, caption="Video Thumbnail",width = "stretch")
                                          st.markdown(
                                                            f"""
                                                        <div style="display:flex; justify-content:center;">
                                                        <div style="width:240px; height:380px; background:#111; border-radius:8px;
                                                                    padding:8px 8px 16px; /* bottom padding */
                                                                    overflow:hidden;">
                                                            <img src="{thumbnail_url}" alt="Video Thumbnail"
                                                                style="width:100%; height:100%; object-fit:cover; border-radius:6px; display:block;"/>
                                                        </div>
                                                        </div>                                                            """,
                                                            unsafe_allow_html=True) 


                                                        

                                          




    
                                    except:pass

                                toggle_label = "🔼 Hide" if st.session_state[show_video_key] else "▶️ Show"
                                if st.button(f"{toggle_label} Preview", key=f"toggle_vid_btn_{grid_instance_key}", help="Show/hide video preview", use_container_width=True):
                                    st.session_state[show_video_key] = not st.session_state[show_video_key]
                                    st.rerun()

                                if st.button("➕ Select (Queue Job)", key=f"select_btn_{grid_instance_key}", type="primary", use_container_width=True, disabled=st.session_state.batch_processing_active):
                                    base_video_id = video_id
                                    base_lang_string = lang_for_group.strip() # Use lang_for_group
                                    langs_to_process = [l.strip() for l in base_lang_string.split(',') if l.strip()]
                                    if not langs_to_process:
                                        langs_to_process = ["default"] # Fallback language
                                        st.warning("Could not parse languages for job, using default.")

                                    for current_lang_for_job in langs_to_process:
                                        base_key_prefix = f"{base_video_id}_{current_lang_for_job}_"
                                        existing_copy_numbers = [
                                            int(k[len(base_key_prefix):])
                                            for k in st.session_state.selected_videos.keys()
                                            if k.startswith(base_key_prefix) and k[len(base_key_prefix):].isdigit()
                                        ]
                                        next_copy_number = max(existing_copy_numbers) + 1 if existing_copy_numbers else 1
                                        job_key = f"{base_key_prefix}{next_copy_number}"

                                        st.session_state.selected_videos[job_key] = {
                                            'Job Key': job_key,
                                            'Search Term': current_search_terms_for_group_display, # Terms used for this result set
                                            'Topic': topic_for_group,
                                            'Language': current_lang_for_job, # Specific language for this job
                                            'Video Title': video_title,
                                            'Video ID': base_video_id,
                                            'Copy Number': next_copy_number,
                                            'Standard URL': standard_video_url,
                                            'fetching_dlp': True,
                                            'Direct URL': None,
                                            'Format Details': None,
                                            'yt_dlp_error': None,
                                            'Generated S3 URL': None,
                                            'Generation Error': None,
                                            'Status': 'Selected, Fetching URL...',
                                            'Script Angle': script_ver_for_group, # Use script_ver_for_group
                                            'BG Music' : bg_music_for_group,    # Use bg_music_for_group
                                            'TTS Voice' : tts_voice_for_group,   # Use tts_voice_for_group
                                            'platform' :platform
                                        }
                                        st.toast(f"Queued Job #{next_copy_number} ({current_lang_for_job}) for: {video_title}", icon="➕")
                                    st.rerun()

                                # --- Display Status for Existing Jobs for THIS video ---
                                related_job_keys = [
                                    k for k, v_data in st.session_state.selected_videos.items()
                                    if v_data.get('Video ID') == video_id and v_data.get('Language') in [l.strip() for l in lang_for_group.split(',') if l.strip()]
                                ]
                                if related_job_keys:
                                    status_expander_key = f"status_expander_{grid_instance_key}"
                                    status_expander = st.expander(f"Show Status for {len(related_job_keys)} Queued Job(s)")
                                    with status_expander:
                                        sorted_job_keys = sorted(related_job_keys, key=lambda k_sort: (st.session_state.selected_videos.get(k_sort, {}).get('Language', ''), st.session_state.selected_videos.get(k_sort, {}).get('Copy Number', 0)))
                                        for r_job_key in sorted_job_keys:
                                            job_data = st.session_state.selected_videos.get(r_job_key)
                                            if job_data:
                                                copy_num = job_data.get('Copy Number', '?')
                                                job_lang = job_data.get('Language', '?')
                                                status = job_data.get('Status', 'Unknown')
                                                s3_url = job_data.get('Generated S3 URL')
                                                error_msg = job_data.get('Generation Error') or job_data.get('yt_dlp_error')
                                                st.markdown(f"**Job #{copy_num} ({job_lang})** (`{r_job_key}`)")
                                                if status == 'Processing': st.info("⚙️ Processing...", icon="⏳")
                                                elif status == 'Queued': st.info("🕒 Queued", icon="🕒")
                                                elif status == 'Completed' and s3_url:
                                                    st.success("✔️ Generated!", icon="🎉")
                                                    st.link_button("View on S3", url=s3_url, type="secondary")
                                                elif status == 'Failed' and error_msg: st.error(f"❌ Failed: {error_msg[:60]}...", icon="🔥")
                                                elif status.startswith('Error:') and error_msg: st.error(f"⚠️ URL Error: {error_msg[:60]}...", icon="⚠️")
                                                elif status == 'Ready': st.success("✅ Ready to Process", icon="👍")
                                                elif status == 'Selected, Fetching URL...': st.info("📡 Fetching URL...", icon="📡")
                                                else: st.write(f"Status: {status}")
            # --- End of Video Grid Display ---

            # --- "Show More" Button (TikTok/Pinterest) ---
            if platform_for_group in ('tk', 'pin') and result_data.get('next_start_index') is not None:
                current_next_start_index = result_data.get('next_start_index')
                requested_batch_size = result_data.get('original_input_count', 10) # Get the batch size for the label
                
                platform_label = "TikTok" if platform_for_group == 'tk' else "Pinterest"
                fetch_fn = search_tiktok_links_google if platform_for_group == 'tk' else search_pinterest_links_google
                cx_id_for_platform = GOOGLE_TIKTOK_CX_ID if platform_for_group == 'tk' else GOOGLE_PINTEREST_CX_ID

                if st.button(f"Show More {requested_batch_size} {platform_label} Results", key=f"show_more_{platform_for_group}_{search_key}", use_container_width=True):
                    if current_next_start_index: # Ensure it's not None one last time
                        # Retrieve necessary data from result_data for the API call
                        original_query_term = result_data.get('original_term')
                        # original_input_count is the total num_results user wants for this topic/search_key
                        total_results_desired_for_topic = result_data.get('original_input_count', 10) 
                        
                        with st.spinner(f"Fetching more {platform_label} results for '{original_query_term}'..."):
                            # current_next_start_index is where the *next block* of results should start
                            new_videos_block, new_block_next_start_index = fetch_fn(
                                api_keys=youtube_api_key_secret,
                                cx_id=cx_id_for_platform,
                                query=original_query_term,
                                num_results=total_results_desired_for_topic, # Ask for another full block
                                start_index=current_next_start_index 
                            )

                            if new_videos_block:
                                st.session_state.api_search_results[search_key]['videos'].extend(new_videos_block)
                                num_newly_fetched = len(new_videos_block)
                                
                                # Update the next_start_index for the *next* "Show More" click
                                st.session_state.api_search_results[search_key]['next_start_index'] = new_block_next_start_index
                                # Update the last_start_index to the one used for *this* fetch
                                st.session_state.api_search_results[search_key]['last_start_index'] = current_next_start_index
                                st.toast(f"Added {num_newly_fetched} more results. Total for this search: {len(st.session_state.api_search_results[search_key]['videos'])}", icon="✅")
                            else: # No videos returned from this "Show More" call
                                st.session_state.api_search_results[search_key]['next_start_index'] = None # No more pages
                                st.toast("No more results found or API error during 'Show More'.", icon="ℹ️")
                            
                        st.rerun()
                    else:
                        # This case should ideally not be reached if button is only shown when current_next_start_index is valid
                        st.warning("Cannot fetch more, start index is invalid.")
            
            # --- "Search More" Logic Placed After Video Grid, Within term_container ---
            st.markdown("---") # Visual separator
            if input_search_term_from_editor.lower() == 'auto' or 1==1:
                if st.button("🔎 Search More Auto Terms", key=f"search_more_btn_auto_{search_key}"):
                    # st.text(input_search_term_from_editor.lower() +'ss')
                    
                        # "Auto Mode": Generate NEW AI terms for the original topic and then search
                        st.info(f"Attempting to generate NEW search terms for 'auto' topic: {topic_for_group}...")
                        new_ai_generated_terms = None
                        try:
                            
                            # Replace with your actual LLM call
                            new_ai_generated_terms = chatGPT(AUTO_TERMS_PROMPT + topic_for_group,client=openai_client) # model="gemini-2.5-flash-preview-04-17"
                            # Or: new_ai_generated_terms = chatGPT(prompt_for_new_terms, client=openai_client)
                            # Or: new_ai_generated_terms = claude(prompt_for_new_terms)

                            if not new_ai_generated_terms or not new_ai_generated_terms.strip():
                                st.error(f"LLM failed to generate new valid terms for '{topic_for_group}'. Original results retained.")
                            else:
                                st.write(f"Newly generated terms for '{topic_for_group}': {new_ai_generated_terms}")
                                # Search YouTube with these NEW terms, using the original count for this topic
                                if platform_for_group == 'yt':
                                    new_videos = search_youtube(youtube_api_key_secret, new_ai_generated_terms, count_from_editor)
                                elif platform_for_group == 'tk':
                                    # Call the refactored search_tiktok_links_google
                                    # It now returns (videos_list, next_block_start_index)
                                    # For a new "auto" search, always start at index 1
                                    videos_list, next_block_start_index = search_tiktok_links_google(
                                        api_keys=youtube_api_key_secret,
                                        cx_id=GOOGLE_TIKTOK_CX_ID,
                                        query=new_ai_generated_terms,
                                        num_results=count_from_editor, # User's desired items per batch
                                        start_index=1 
                                    )
                                    new_videos = videos_list # Keep 'new_videos' name for the common logic below
                                elif platform_for_group == 'pin':
                                    videos_list, next_block_start_index = search_pinterest_links_google(
                                        api_keys=youtube_api_key_secret,
                                        cx_id=GOOGLE_PINTEREST_CX_ID,
                                        query=new_ai_generated_terms,
                                        num_results=count_from_editor,
                                        start_index=1
                                    )
                                    new_videos = videos_list

                                if new_videos is not None: 
                                    st.session_state.api_search_results[search_key]['videos'] = new_videos # Replace with new results
                                    st.session_state.api_search_results[search_key]['original_term'] = new_ai_generated_terms
                                    if platform_for_group in ('tk', 'pin'): # Specific updates for pagination platforms
                                        st.session_state.api_search_results[search_key]['next_start_index'] = next_block_start_index
                                        st.session_state.api_search_results[search_key]['last_start_index'] = 1
                                    # For YouTube, 'next_start_index' etc. are not currently used in this 'auto search more' block.
                                    # If YT were to have a similar "show more blocks" feature, this would need adjustment.

                                    st.toast(f"Fetched new 'auto' results for '{topic_for_group}' using fresh AI terms.", icon="🔄")
                                else: # new_videos is None, indicating critical API error from search function
                                    st.toast(f"Failed to fetch new 'auto' results for '{topic_for_group}'. Search API error.", icon="⚠️")
                                st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred while generating new terms or searching: {e}")
                            # Not rerunning here to allow user to see the error
                if st.button("🔎 Search More Manual Terms", key=f"search_more_btn_manual_{search_key}"):
                 # Original input_search_term was not 'auto', so this is for manual refinement
                    st.session_state.search_more_manual_input_visible[search_key] = not st.session_state.search_more_manual_input_visible[search_key]
                    if not st.session_state.search_more_manual_input_visible[search_key]: # If hiding, clear query
                        st.session_state.search_more_manual_query[search_key] = ""
                    st.rerun()

            # --- Manual input section (conditionally displayed) ---
            if st.session_state.search_more_manual_input_visible.get(search_key, False): 
                cols_manual = st.columns([0.7, 0.3]) # Adjust column ratios as needed
                with cols_manual[0]:
                    st.session_state.search_more_manual_query[search_key] = st.text_input(
                        f"Enter new search term for topic '{topic_for_group}':",
                        value=st.session_state.search_more_manual_query.get(search_key, ""),
                        key=f"manual_query_text_{search_key}",
                        placeholder="e.g., amazing gadget reviews"
                    )
                with cols_manual[1]:
                    # Add a little space above the button for alignment if needed
                    st.write("") # Or use st.markdown("<br>", unsafe_allow_html=True) for more control
                    if st.button("Search with New Term", key=f"submit_manual_query_{search_key}", use_container_width=True):
                        new_manual_term = st.session_state.search_more_manual_query.get(search_key, "").strip()
                        if new_manual_term:
                            st.info(f"Searching with new term '{new_manual_term}' for topic: {topic_for_group}...")
                            # Search YouTube with the new manual term, using the original count for this topic
                            if platform_for_group == 'yt':
                                new_videos = search_youtube(youtube_api_key_secret, new_manual_term, count_from_editor)
                            elif platform_for_group == 'tk':
                                # Call the refactored search_tiktok_links_google
                                # For a new manual search, always start at index 1
                                videos_list, next_block_start_index = search_tiktok_links_google(
                                    api_keys=youtube_api_key_secret,
                                    cx_id=GOOGLE_TIKTOK_CX_ID,
                                    query=new_manual_term,
                                    num_results=count_from_editor, # User's desired items per batch
                                    start_index=1
                                )
                                new_videos = videos_list # Keep 'new_videos' name for common logic
                            elif platform_for_group == 'pin':
                                videos_list, next_block_start_index = search_pinterest_links_google(
                                    api_keys=youtube_api_key_secret,
                                    cx_id=GOOGLE_PINTEREST_CX_ID,
                                    query=new_manual_term,
                                    num_results=count_from_editor,
                                    start_index=1
                                )
                                new_videos = videos_list

                            if new_videos is not None:
                                st.session_state.api_search_results[search_key]['videos'] = new_videos # Replace with new results
                                st.session_state.api_search_results[search_key]['original_term'] = new_manual_term
                                if platform_for_group in ('tk', 'pin'): # Specific updates for pagination
                                    st.session_state.api_search_results[search_key]['next_start_index'] = next_block_start_index
                                    st.session_state.api_search_results[search_key]['last_start_index'] = 1
                                # Similar to 'auto search more', YT doesn't use these pagination fields in this block yet.
                                st.toast(f"Updated results for '{topic_for_group}' with new term.", icon="🔄")
                            else: # new_videos is None
                                st.toast(f"Search with new term failed for '{topic_for_group}'. API error.", icon="⚠️")

                            st.session_state.search_more_manual_input_visible[search_key] = False # Hide after submit
                            st.session_state.search_more_manual_query[search_key] = "" # Clear the input
                            st.rerun()
                        else:
                            st.warning("Please enter a search term for manual refinement.")
            # --- "Search More" Logic END ---
        # --- End of term_container ---

else: # No api_search_results yet
    if 'search_triggered' in st.session_state and st.session_state.search_triggered == False and not st.session_state.api_search_results:
         # This condition might be true if a search was run but yielded absolutely nothing for any input row.
        st.info("The previous search returned no results for any of the topics/terms.")
    else: # Initial state or after clearing
        st.info("Perform a search using the sidebar to see video results here.")


# The rest of the script (yt-dlp fetching, batch processing, sidebar display) follows...

# 4. yt-dlp Fetching Logic (Runs after UI render if needed)
# (Use the corrected version that works with job_keys and caches results)
if not st.session_state.batch_processing_active:



    job_keys_to_fetch = [
        job_key for job_key, data in st.session_state.selected_videos.items()
        if data.get('fetching_dlp') # Check the flag
    ]

    if job_keys_to_fetch:
        fetch_job_key = job_keys_to_fetch[0] # Process one per rerun cycle
        video_data = st.session_state.selected_videos.get(fetch_job_key)

        if video_data:
            standard_url = video_data.get('Standard URL')
            title = video_data.get('Video Title', fetch_job_key)
            platform = video_data.get('platform')


            if platform in ('tk','pin'):
                current_state = st.session_state.selected_videos.get(fetch_job_key)
                current_state['fetching_dlp'] = False
                current_state['Direct URL'] = standard_url
                current_state['Status'] = 'Ready'



            elif platform == 'yt':
                # Only show spinner if fetching is needed
                with st.spinner(f"Fetching video details for '{title}' (Job: {fetch_job_key} {standard_url})..."):
                    dlp_info = None
                    # Check cache first using the standard watch URL

                    


                    if standard_url and standard_url in st.session_state.get('resolved_vid_urls', {}):
                        dlp_info = st.session_state['resolved_vid_urls'][standard_url]
                        logging.info(f"Cache hit for {standard_url}") # Debug
                    elif standard_url:
                        logging.info(f"Cache miss - Fetching NEW URL info for {standard_url}") # Debug
                        # Fetch info using yt-dlp helper function
                        dlp_info = get_yt_dlp_info(standard_url)
                        # Update cache ONLY if fetch was successful AND returned a direct URL
                        if dlp_info and dlp_info.get('direct_url'):
                            st.session_state.setdefault('resolved_vid_urls', {})[standard_url] = dlp_info
                    else:
                        dlp_info = {'error': 'Missing Standard URL in job data.'} # Cannot fetch without URL

                # Update state for the specific job_key
                current_state = st.session_state.selected_videos.get(fetch_job_key)
                if current_state:
                    current_state['fetching_dlp'] = False # Mark fetch attempt as done

                    if dlp_info and dlp_info.get('direct_url'):
                        current_state['Direct URL'] = dlp_info['direct_url'] # Store the direct URL
                        current_state['Format Details'] = dlp_info.get('format_details', 'N/A')
                        current_state['yt_dlp_error'] = None
                        current_state['Status'] = 'Ready' # Ready for processing
                        st.toast(f"Direct URL loaded for job '{fetch_job_key}'", icon="✅")
                    else: # Handle errors or missing URL from dlp_info
                        error_detail = dlp_info.get('error', "Could not get direct URL") if dlp_info else "yt-dlp fetch failed critically"
                        current_state['Direct URL'] = None
                        current_state['Format Details'] = "Error"
                        current_state['yt_dlp_error'] = error_detail
                        current_state['Status'] = f"Error: {error_detail}" # Update status to reflect error
                        st.toast(f"yt-dlp failed for job '{fetch_job_key}': {error_detail}", icon="⚠️")

            # Save updated state
            st.session_state.selected_videos[fetch_job_key] = current_state
            st.rerun() # Rerun to update UI display


# 5. Video Generation Logic (BATCH PROCESSING)
# (Use the corrected version that processes one job_key at a time)
if st.session_state.batch_processing_active and st.session_state.generation_queue:
    job_key_to_process = st.session_state.generation_queue[0]
    video_data = st.session_state.selected_videos.get(job_key_to_process)
    final_video_path = None # For cleanup
    audio_path = None       # For cleanup

    # Check if job data exists and is ready (has Direct URL, no prior error)
    if video_data and video_data.get('Direct URL') and not video_data.get('yt_dlp_error'):
        processed_count_display = st.session_state.batch_processed_count + 1
        total_count_display = st.session_state.batch_total_count
        st.header(f"⚙️ Processing Job {processed_count_display}/{total_count_display}: {video_data['Video Title']} (Copy #{video_data.get('Copy Number', '?')} platform {video_data.get('platform' , 'na')})")
        gen_placeholder = st.container() # Container for logs of this specific job run

        try:
            # --- Update Status ---
            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Processing'

            # --- Generation Steps (run once per job key) ---
            with gen_placeholder:
                st.info(f"Starting video generation process for job: {job_key_to_process}")
                with st.status("Running generation steps...", state="running", expanded=True) as status_log:
                    try:
                        # --- 1. Get Job Data ---
                        topic = video_data.get('Topic', 'video topic')
                        lang = video_data.get('Language', 'English')
                        script_ver = video_data.get("Script Angle")
                        bg_music = video_data.get('BG Music', False)
                        tts_voice = video_data.get('TTS Voice', 'sage')
                        base_video_direct_url = video_data.get("Direct URL") # Use the fetched direct URL
                        copy_num = video_data.get('Copy Number', 0),
                        platform = video_data.get('platform' , "na")

                        if not base_video_direct_url:
                            raise ValueError("Direct video URL missing.")

                        # --- 2. Generate Script ---
                        st.write(f"1/5: Generating script (Angle: {script_ver})...")
                        if script_ver == "mix":
                            script_ver_temp = random.choice([opt for opt in SCRIPT_VER_OPTIONS if opt != 'mix'])
                        if ',' in script_ver:
                            script_ver_temp = random.choice(script_ver.split(","))
                        else:
                            script_ver_temp = script_ver
                        # --- Construct the full script prompt based on script_ver_temp ---
                        # (Insert your actual prompt logic here, using f-strings)
                        if  script_ver_temp == 'default_v2' :
                            

                            script_prompt = f"""Generate a short voiceover script (approx. 15-20 seconds, typically 2-3 concise sentences) for a social media video ad about '{topic}' in {lang}.

**Goal:** Create an intriguing and engaging script that captures attention quickly, holds it (retentive), and encourages clicks, suitable for platforms like Facebook/Instagram Reels/TikTok.

**Tone:** Conversational, casual, and authentic. Avoid sounding like a formal advertisement or overly 'salesy'. Speak *to* the viewer directly (use "you" where appropriate).

**Key Requirements:**
1.  **Strong Hook:** Start immediately with something captivating (e.g., a relatable question, a surprising statement, the core benefit) to grab attention in the first 1-2 seconds.
2.  **Concise Body:** Briefly convey the most interesting or beneficial aspect of the '{topic}'. Focus on clarity and smooth flow.
3.  **Clear Call to Action (CTA):** End the script *only* with the phrase "Tap now to " followed by a simple, clear, non-committal action (e.g., learn more, see details, explore options, find out how).

**Strict Exclusions (Mandatory):**
* **NO:** "we," "our," or "I."
* **NO:** Sensational language, hype, exaggeration, or false promises. (Be truthful and grounded).
* **NO:** Aggressive or fake urgency/scarcity tactics (e.g., "Act fast!", "Limited spots!").
* **NO:** Geographically suggestive terms (e.g., "Near you," "In your area").
* **NO:** Jargon or overly complex vocabulary.
* **NO:** DONT make false promises.
NO ('get approved') 'See what's available near you' ' 'available this weekend\month' etc!!!

**Output:** Provide ONLY the raw script text, with no extra explanations or formatting.  """

                        elif 'v3' in lang:
                            script_prompt = f"""
You are an expert scriptwriter for high-performing short-form video ads. Generate a voiceover script based on the following parameters:

**Topic:** {topic}
**Language:** {lang}
**Target Length:** 15-20 seconds (2-3 very concise sentences).
**Platform Context:** Viral-style social media ad (e.g., Reels, TikTok).

**Choose ONE Core Angle for the script:**
* **Problem/Solution:** Briefly identify a relatable pain point and position the '{topic}' as the clear solution.
* **Transformation/Benefit:** Focus entirely on the positive outcome, feeling, or change the '{topic}' enables for the viewer.
* **Intrigue/Curiosity:** Hint at something unique, surprising, or valuable about the '{topic}' without giving everything away, making the viewer want to know more.

**Script Structure & Tone:**
1.  **Hook (1-2 seconds):** Start *immediately* with the most compelling part of your chosen angle (the problem, the benefit, the surprising fact). Must grab attention instantly.
2.  **Body (Concise):** Briefly elaborate on the hook, sticking to the chosen angle. Use simple, direct, conversational language. Address the viewer using "you." Maintain an authentic, non-hypey tone.
3.  **Call to Action (CTA - Final words):** Conclude *only* with "Tap now to " followed by a low-commitment action phrase (e.g., learn more, see details, explore, find out why).

**Mandatory Constraints:**
* Exclude: "we," "our," "I."
* Exclude: Hype, false promises, sensationalism.
* Exclude: Fake urgency or scarcity.
* Exclude: Geographic limitations.
* Exclude: Complex jargon. Keep words simple.
CTA in the likes of: 'Click to explore options or 'Tap to see how it works.'.but still highly engaging high CTR not generic. dont use 'to see models\what's available ... etc'
NO ('get approved') 'See what's available near you' ' 'available this weekend\month' etc!!!
**Output Requirement:** Return *only* the final script text.
                            



"""

                        elif script_ver_temp == "default":
                            script_prompt = f"""Create a short, engaging voiceover script for FB viral   video (roughly 15-20 seconds long, maybe 2-3 sentences) about '{topic}' in language {lang}. The tone should be informative yet conversational, '.  smooth flow. Just provide the script text, nothing else. create intriguing and engaging script, sell the topic to the audience . be very causal and not 'advertisement' style vibe. end with a call to action 'tap to....'  .the text needs to be retentive.Don't say 'we' or 'our' .NOTE:: DO NOT dont use senetional words and phrasing and DONT make false promises , use Urgency Language, Avoid geographically suggestive terms (e.g., "Near you," "In your area"). Do not use "we" or "our". end with CTA in the likes of: 'Click to explore options or 'Tap to see how it works.'  or similar!!!!  but still highly engaging high CTR not generic and that pushes value and benefit to the viewer and convice to click . dont use 'to see models\what's available ... etc'. in end if video use something "Tap now to.." with a clear, non-committal phrase !!! NO ('get approved') 'See what's available near you' ' 'available this weekend\month' etc!!!  """
                        # script_text = chatGPT(script_prompt,model="o1", client=openai_client)
                        elif script_ver_temp == "default_clean":
                            script_prompt = f"""Create a short, engaging voiceover script for FB viral   video (roughly 15-20 seconds long, maybe 2-3 sentences) about '{topic}' in language {lang}. The tone should be informative yet conversational, '.  smooth flow. Just provide the script text, nothing else. create intriguing and engaging script, sell the topic to the audience . be very causal and not 'advertisement' style vibe. end with a call to action in appropriate lang in the likes: 'Read More about .. '   .the text needs to be retentive.Don't say 'we' or 'our' .NOTE:: DO NOT  use senetional words and phrasing and DONT make false promises , dont use Urgency Language like 'today' or 'immediately',  Avoid geographically suggestive terms (e.g., "Near you," "In your area"). Do not use "we" or "our".  but still highly engaging high CTR not generic and that pushes value and benefit to the viewer and convice to click . dont use 'to see models\what's available ... etc'.\nThe ad must describe the article (the guide/tips), not the service (the loan/job/bike), Treat the user as a reader
                                \n  \n  NO 'get approved', 'See what's available near you' ', 'available this' weekend\month'  promises claims deals . .no Urgency\miracle phrasing etc!!!  """
                        # script_text = chatGPT(script_prompt,model="o1", client=openai_client)


                        elif script_ver_temp == '1st_person':
                            script_prompt = f"""
                                            Create a brief, captivating first-person voiceover script for a viral FB video about '{topic}' in {lang}. 
                                            Keep it concise (15-20 seconds when spoken, about 2-3 sentences) with these guidelines:

                                            - Start with an immediate hook in the first 3-5 seconds to grab attention
                                            - The hook should be intriguing but honest - NO false promises or misleading claims
                                            - Use first-person perspective throughout
                                            - Make the tone authentic and conversational, like a friend sharing a discovery
                                            - Focus on creating genuine interest in the topic with real value
                                            - Maintain a natural flow that keeps viewers watching
                                            - End with a simple call to action like "Tap to discover..." or "Tap to learn..."
                                            - Ensure the content feels genuine, not like an advertisement

                                            IMPORTANT:
                                            - The opening hook must be attention-grabbing AND truthful
                                            - Avoid sensational language or exaggerated claims
                                            - Don't make promises that can't be delivered
                                            - No urgency phrases like "limited time" or "act now"
                                            - No geographic claims (e.g., "near you," "in your area")
                                            - No "we" or "our" language - keep it personal
                                            - End with "Tap to..." followed by a clear, non-committal action
                                            * **NO:** DONT make false promises.NO ('get approved') 'See what's available near you' ' 'available this weekend\month' etc!!!
                                            end with CTA in the likes of: 'Click to explore options or 'Tap to see how it works.' or similar!!!! but highly engaging high CTR not generic.. dont use 'to see models\what's available ... etc'
                                            Return only the script text itself, nothing else.
                                            """


                        st.text(f"using script: {script_ver_temp}")

                        # --- Choose LLM ---
                        # script_text = chatGPT(script_prompt, client=openai_client)
                        script_text  = claude(script_prompt,is_thinking=True) # Assumes claude function uses API key from secrets

                        if not script_text: raise ValueError("Failed to generate script text.")
                        st.text_area("Generated Script:", script_text, height=100, disabled=True, key=f"script_{job_key_to_process}")


                         # --- 3. select voice id ---



                        # --- 3. Generate TTS ---
                        if ',' in tts_voice:
                            tts_voice_temp = random.choice(tts_voice.split(","))
                        else: tts_voice_temp = tts_voice
                        st.write(f"2/5: Generating TTS audio & timestamps...")
                        audio_path, word_timings = generate_audio_with_timestamps(
                            script_text, client=openai_client, voice_id=tts_voice_temp
                        )
                        if audio_path is None or word_timings is None: # Check both for failure
                            raise ValueError("Failed to generate TTS audio or timestamps.")

                        # --- 4. Process Video ---
                        st.write(f"3/5: Processing base video & adding audio/subtitles...")
                        current_with_music = bg_music
                        if current_with_music == 'mix': current_with_music = random.choice(BG_VER_OPTIONS)

                        # Pass Direct URL and other necessary data
                        # This function now downloads the direct url, processes, and returns temp output path

                        
                        final_video_path = process_video_with_tts(
                            base_video_url=base_video_direct_url, # Pass the direct URL
                            audio_path=audio_path,
                            word_timings=word_timings,
                            topic=topic,
                            lang=lang,
                            copy_num=copy_num,
                            with_music=current_with_music,
                            platform= platform
                        )
                        if not final_video_path: raise ValueError("Video processing (MoviePy) failed.")

                        # --- 5. Construct Unique S3 Filename ---
                        safe_topic = urllib.parse.quote(topic.replace(' ', '_')[:30], safe='')
                        timestamp = int(datetime.datetime.now().timestamp())
                        base_video_slug = urllib.parse.quote(base_video_direct_url, safe='')
                        final_s3_object_name = f"{S3_VIDSLIDE_PREFIX}/final_{safe_topic}_{lang}_copy{copy_num}_{timestamp}_base_{base_video_slug}.mp4"
                        # --- 6. Upload to S3 ---
                        st.write(f"4/5: Uploading '{final_s3_object_name}' to S3...")
                        s3_url = upload_vid_to_s3(
                            s3_cli=s3_client, video_path=final_video_path,
                            bucket_name=s3_bucket_name, object_name=final_s3_object_name,
                            region_name=s3_region, endpoint_url=s3_endpoint_url
                        )
                        if not s3_url: raise ValueError("Failed to upload video to S3.")

                        # --- 7. Success ---
                        st.write(f"5/5: Generation Complete!")
                        status_log.update(label="Generation Complete!", state="complete", expanded=False)
                        if job_key_to_process in st.session_state.selected_videos:
                            st.session_state.selected_videos[job_key_to_process]['Generated S3 URL'] = s3_url
                            st.session_state.selected_videos[job_key_to_process]['Generation Error'] = None
                            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Completed'
                            st.success(f"✅ Job '{job_key_to_process}' completed!", icon="🎉")
                            st.video(s3_url) # Show the final video
                        else: st.warning(f"Job key {job_key_to_process} missing after completion.")

                    # --- Error Handling within Status Log ---
                    except Exception as e:
                        st.error(f"Error processing job '{job_key_to_process}': {e}", icon="🔥")
                        status_log.update(label=f"Generation Failed: {str(e)[:100]}", state="error", expanded=True)
                        if job_key_to_process in st.session_state.selected_videos:
                            st.session_state.selected_videos[job_key_to_process]['Generation Error'] = str(e)[:200]
                            st.session_state.selected_videos[job_key_to_process]['Generated S3 URL'] = None
                            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Failed'
                        # Allow finally block to handle queue/rerun

        # --- Finally block for cleanup and queue management ---
        finally:
            st.write(f"--- Cleaning up temporary files for job {job_key_to_process} ---")
            # Clean up TTS audio file
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except Exception as rm_err: st.warning(f"Could not delete temp audio: {audio_path} ({rm_err})")
            # Clean up final processed video temp file (returned by process_video_with_tts)
            if final_video_path and os.path.exists(final_video_path):
                try: os.remove(final_video_path)
                except Exception as rm_err: st.warning(f"Could not delete final temp video: {final_video_path} ({rm_err})")
            # Note: process_video_with_tts handles its internal temp files (downloaded base, moviepy temps)

            # Manage Queue
            if st.session_state.generation_queue and st.session_state.generation_queue[0] == job_key_to_process:
                 st.session_state.generation_queue.pop(0)
                 st.session_state.batch_processed_count += 1
            elif st.session_state.generation_queue: # Log unexpected queue state
                 st.warning(f"Queue state unexpected. Expected {job_key_to_process}, found {st.session_state.generation_queue[0]}. Popping front.")
                 st.session_state.generation_queue.pop(0)
                 st.session_state.batch_processed_count += 1

            # Check if batch is finished
            if not st.session_state.generation_queue:
                st.session_state.batch_processing_active = False
                st.balloons()
                st.success("🎉 Batch processing finished!")
                if not st.session_state.get("sheet_appended", False):
                    input_reference_df = st.session_state.get('current_search_df')
                    if input_reference_df is None or input_reference_df.empty:
                        input_reference_df = st.session_state.get('search_data')
                    df_topic_summary = create_topic_summary_dataframe(
                        st.session_state.selected_videos,
                        input_reference_df
                    )
                    if not df_topic_summary.empty:
                        google_sheets_append_df(
                            "13TOgYTYpVV0ysvKqufS2Q5RDdgTP097x1hH_eMtCL4w",
                            "yt_video!A1",
                            df_topic_summary
                        )
                    st.session_state.sheet_appended = True


            # Rerun to process next item or update UI
            st.rerun()

    # --- Logic for skipping invalid jobs ---
    elif job_key_to_process in st.session_state.selected_videos:
         st.warning(f"Skipping job {job_key_to_process}. Invalid Direct URL or previous error.", icon="❓")
         # Update status to Skipped
         st.session_state.selected_videos[job_key_to_process]['Status'] = 'Skipped (Invalid Data/URL)'
         st.session_state.selected_videos[job_key_to_process]['Generation Error'] = 'Skipped - Invalid data or URL before processing'
         # Manage Queue
         if st.session_state.generation_queue: st.session_state.generation_queue.pop(0)
         st.session_state.batch_processed_count += 1
         if not st.session_state.generation_queue: st.session_state.batch_processing_active = False
         st.rerun()
    else: # Job key somehow disappeared
        st.error(f"Job key {job_key_to_process} in queue but not found. Removing.", icon="❓")
        if st.session_state.generation_queue: st.session_state.generation_queue.pop(0)
        st.session_state.batch_processed_count += 1
        if not st.session_state.generation_queue: st.session_state.batch_processing_active = False
        st.rerun()


# --- Display Selected Jobs Table & Summary (Sidebar) ---
# (Use the corrected display logic from the previous response)
st.sidebar.divider()
st.sidebar.header("Selected & Generated Video Jobs")

if 'selected_videos' in st.session_state and st.session_state.selected_videos:
    selected_list = list(st.session_state.selected_videos.values())
    if selected_list:
        df_selected = pd.DataFrame(selected_list)

        # Define columns for the detailed status table
        display_columns = [
            'Status', 'Video Title', 'Copy Number', 'Topic','Language', 'Script Angle',
            'Video ID', #'Search Term', # Optional
            #'Format Details', # Often too verbose
            'yt_dlp_error', 'Generation Error',
            'Generated S3 URL', 'Job Key'
        ]
        # Ensure required columns exist, fill missing with NA
        for col in display_columns:
             if col not in df_selected.columns:
                  df_selected[col] = pd.NA

        # Clean up display values
        df_selected['Status'] = df_selected['Status'].fillna('Unknown')
        df_selected['Copy Number'] = df_selected['Copy Number'].fillna('?')
        df_selected['yt_dlp_error'] = df_selected['yt_dlp_error'].fillna('OK') # Display OK if no error
        df_selected['Generation Error'] = df_selected['Generation Error'].fillna('OK') # Display OK if no error

        # Create the display DataFrame with selected columns and sorting
        df_selected_display = df_selected[display_columns].copy()
        # Sort by Job Key or other columns as desired
        try: # Sort by Copy Number within Video ID/Lang if possible
             df_selected_display.sort_values(by=['Video ID', 'Language', 'Copy Number'], inplace=True)
        except: # Fallback sort
             df_selected_display.sort_values(by='Job Key', inplace=True)


        # Display the detailed DataFrame
        st.sidebar.dataframe(
            df_selected_display,
            column_config={
                 "Generated S3 URL": st.column_config.LinkColumn("S3 Link"),
                 "yt_dlp_error": st.column_config.TextColumn("URL Fetch Status", width="small", help="Status of fetching direct video URL"),
                 "Generation Error": st.column_config.TextColumn("Generation Status", width="small", help="Status of the video generation process"),
                 "Copy Number": st.column_config.NumberColumn("Copy #", width="small"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Download Button for detailed status
        # try:
        #     csv_data = convert_df_to_csv(df_selected_display) # Use cached conversion
        #     st.sidebar.download_button(
        #          label="📥 Download Job Status (CSV)",
        #          data=csv_data,
        #          file_name='video_generation_job_status.csv',
        #          mime='text/csv',
        #          use_container_width=True,
        #          disabled=st.session_state.batch_processing_active
        #     )
        # except Exception as e:
        #     st.sidebar.warning(f"Could not generate detailed CSV: {e}")

        # --- Topic Summary DataFrame Section ---
        st.sidebar.divider()
        st.sidebar.subheader("Generated Video Summary by Topic")
        input_reference_df = st.session_state.get('current_search_df')
        if input_reference_df is None or input_reference_df.empty:
            input_reference_df = st.session_state.get('search_data')
        df_topic_summary = create_topic_summary_dataframe(
            st.session_state.selected_videos,
            input_reference_df
        )

        if not df_topic_summary.empty:
            st.sidebar.dataframe(
                 df_topic_summary,
                 use_container_width=True,
                 # Dynamically create LinkColumn config for vidX_url columns
                 column_config={
                     col: st.column_config.LinkColumn(f"Video {i+1}", display_text="View")
                     for i, col in enumerate(df_topic_summary.columns) if col.startswith('vid') and col.endswith('_url')
                 },
                 hide_index=True
            )
            # google_sheets_append_df("13TOgYTYpVV0ysvKqufS2Q5RDdgTP097x1hH_eMtCL4w","yt_video!A1",df_topic_summary  )

            # Download Button for summary
            try:
                csv_summary_data = convert_df_to_csv(df_topic_summary)
                st.sidebar.download_button(
                    label="📥 Download Topic Summary (CSV)",
                    data=csv_summary_data,
                    file_name='generated_videos_by_topic.csv',
                    mime='text/csv',
                    use_container_width=True,
                    disabled=st.session_state.batch_processing_active
                )
            except Exception as e:
                st.sidebar.warning(f"Could not generate summary CSV: {e}")
        else:
            st.sidebar.info("No videos successfully generated yet for summary.")

    else: # If selected_list is empty
         st.sidebar.info("No video jobs selected yet.")
else: # If selected_videos dict is empty or doesn't exist
    st.sidebar.info("No video jobs selected yet.")

# Footer notes
st.sidebar.caption("Each 'Select' click queues one job. URL Fetch after selection. Video Gen uses direct URL.")
