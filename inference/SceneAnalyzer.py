import re
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import google.generativeai as genai




class SceneAnalyzer:

    def __init__(self, api_key="AIzaSyDCdXM9KLj4SgalJU3_vRYRAZEJNwhPfWY", model_name='gemini-2.5-flash'):
        """
        Initialize the scene analyzer with Gemini model.
        
        Parameters:
            api_key (str): Google Generative AI API key
            model_name (str): Name of the Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.zero_shot_prompt = """
        You are a scene analyst. Based on the metadata below, classify the scene into one category from: 
        Conflict, Dialogue, Reunion, Chase, Celebration, Romantic, Solitude, Tense, Casual, Danger, Mourning, Relaxation. 
        Then, write a two-line emotionally descriptive caption focusing on facial expressions, voice tone, 
        body language, and overall mood.
        Metadata:
        Objects: {objects}
        Face Emotion: {face_emotion}
        Caption: "{caption}"
        Transcript: "{transcript}"
        Voice Emotion: {voice_emotion}
        Tone Level: {tone_level}
        Category:
        Description:
        """.strip()

    def predict_scene(self, caption, objects, face_emotion, speech, voice_emotion, tone_level):
        """
        Predict high-level scene category and generate description.
        
        Parameters:
            caption (str): Image caption
            objects (list): Detected objects in scene
            face_emotion (str): Predicted face emotion
            speech (str): Transcript of speech
            voice_emotion (str): Predicted voice emotion
            tone_level (str): Speech tone level
            
        Returns:
            str: Formatted response with category and description
        """
        prompt = self.zero_shot_prompt.format(
            caption=caption,
            objects=objects,
            face_emotion=face_emotion,
            transcript=speech if speech.strip() else "no text",
            voice_emotion=voice_emotion,
            tone_level=tone_level
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating scene analysis: {e}")
            return "Analysis error"

    def analyze_scenes(self, scenes_df, show_images=False):
        """
        Analyze all scenes in dataframe and add rich descriptions.
        
        Parameters:
            scenes_df (pd.DataFrame): DataFrame containing scene metadata
            show_images (bool): Whether to display images with analysis
            
        Returns:
            pd.DataFrame: Modified dataframe with analysis columns added
        """
        scenes_df = scenes_df.copy()
        
        # Initialize columns
        scenes_df['scene_category'] = ""
        scenes_df['scene_description'] = ""
        
        for idx, row in tqdm(scenes_df.iterrows(), total=scenes_df.shape[0]):
            result = self.predict_scene(
                caption=row.get('caption', ''),
                objects=row.get('objects', []),
                face_emotion=row.get('face_emotion', ''),
                speech=row.get('transcript', ''),
                voice_emotion=row.get('voice_emotion', ''),
                tone_level=row.get('tone_level', '')
            )
            
            # Parse the response
            if "\n" in result:
                category, description = result.split("\n", 1)
                category = category.replace("Category:", "").strip()
                description = description.replace("Description:", "").strip()
            else:
                category = "Unknown"
                description = result
                
            scenes_df.at[idx, 'scene_category'] = category
            scenes_df.at[idx, 'scene_description'] = description
            
            if show_images and 'path' in row:
                img = cv2.imread(row['path'])
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img_rgb)
                    plt.axis('off')
                    title = f"Scene: {row.get('filename', '')}\nCategory: {category}"
                    plt.title(title)
                    plt.show()
                else:
                    print(f"Image not found at {row['path']}")
                    
        return scenes_df