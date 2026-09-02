import asyncio
import pandas as pd
from eksipy import Eksi
import re
from langdetect import detect
from tqdm import tqdm  
from openai import OpenAI
import random


topic_names = [
    #Tech / Programming / AI
    "spotify", "google", "python", "ai", "chatgpt", "openai", "midjourney", "dalle", 
    "javascript", "react", "nodejs", "git", "blockchain", "web3", "crypto", "nft", "startup", "venture capital",

    #Social Media / Internet Culture
    "facebook", "instagram", "twitter", "youtube", "tiktok", "linkedin", "discord", "reddit", 
    "influencer", "meme", "shitpost", "gaming", "twitch", "streamer",

    #Entertainment / Gaming / Anime
    "netflix", "prime video", "disney+", "minecraft", "lol", "anime", "kpop", "manga", "webtoon", 
    "rap", "hiphop", "edm", "dj",

    #Sports
    "futbol", "basketbol", "tennis", "formula 1", "voleybol", "gym", "workout", "fitness",

    #Lifestyle / Travel / Food / Shopping
    "yemek", "tatil", "seyahat", "alışveriş", "fashion", "sneakers", "nike", "adidas", 
    "apple", "samsung", "mobile", "app", "airbnb", "hotel", "vegan", "fitness lifestyle",

    #Daily Life / Work / Education
    "üniversite", "iş", "freelance", "marketing", "growth", "seo", "branding", "startup life", 
    "gündem", "aşk", "sağlık", "covid", "ekonomi", "politika", "haberler", "finance",

    #Misc / Fun / Pop Culture
    "lol memes", "viral", "challenge", "trend", "gaming memes", "internet slang", 
    "funny videos", "edutainment", "tech news", "apps review", "music", "concert", "festival"
]


async def fetch_and_save_entries(topic_names, output_file="eksi_entries_200pgs.csv"):
    eksi = Eksi()
    all_data = []

    for topic_name in topic_names:
        print(f"Fetching entries for topic: {topic_name}")
        try:
            topic = await eksi.getTopic(topic_name)
            max_pages = 200  #Increase pages as needed
            for page in range(1, max_pages + 1):
                try:
                    entries = await topic.getEntrys(page=page)
                    if not entries:
                        break

                    for entry in entries:
                        content = entry.entry
                        text_str = content.text() if hasattr(content, 'text') else str(content)
                        all_data.append({
                            "topic": topic_name,
                            "page": page,
                            "entry": text_str
                        })

                    #Save periodically
                    if len(all_data) >= 200:
                        df = pd.DataFrame(all_data)
                        df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
                        all_data = []

                    await asyncio.sleep(1)

                except Exception as e:
                    print(f"Error on page {page} of {topic_name}: {e}")
                    break

            await asyncio.sleep(random.uniform(1.5, 3.0))

        except Exception as e:
            print(f"Error fetching topic {topic_name}: {e}")
            await asyncio.sleep(random.uniform(3.0, 6.0))

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    print("Done fetching all entries!")

#***Langdetect***#

def is_code_mixed_langdetect(text):
    """
    Detects whether text contains both Turkish ('tr') and English ('en') words.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    words = re.findall(r'\b\w+\b', text)
    langs = set()

    for word in words:
        try:
            langs.add(detect(word))
        except:
            continue

    return 'en' in langs and 'tr' in langs


#***Filtering Pipeline****#

def filter_code_mixed_entries_df(df, detection_func, output_file="eksi_code_mixed.csv"):
    """
    Detects code-mixed posts from a DataFrame and saves the filtered subset.
    Displays a progress bar.
    """
    print(f"Total entries loaded: {len(df)}")

    #tqdm progress bar integration
    tqdm.pandas(desc="Detecting code-mixed entries")

    #Apply detection function
    df['is_code_mixed'] = df['entry'].progress_apply(lambda x: detection_func(x))

    #Filter for True
    filtered_df = df[df['is_code_mixed'] == True]

    #Save filtered results
    filtered_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(filtered_df)} code-mixed entries to {output_file}")

    #Summary per topic
    summary = filtered_df.groupby('topic').size().reset_index(name='code_mixed_count')
    print("\nCode-mixed counts per topic:")
    print(summary)

    return filtered_df

def chatgpt_prompt_1(posts_batch):
    posts_text = "\n".join(f"{i+1}. {post}" for i, post in enumerate(posts_batch))
    prompt = f"""
        Detect if the given Turkish social media post contains any English words. If English words are present, 
        return true.
        
        Steps 
        1. Language Detection: 
        Detect whether the given text contains English words. 
        
        Output Format
        Detection Result: Output "True" if English words are present, otherwise "False". 
        Number your answers to match the post numbers, and put each response on its own line.

        Example output:
        1. True
        2. False
        etc

        Posts:
        {posts_text}
    """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )

    #Parse numbered output lines
    lines = [line.strip() for line in response.output_text.strip().split("\n") if line.strip()]

    results = []
    for i, post in enumerate(posts_batch):
        #Find the corresponding line starting with the post number
        match_line = next((line for line in lines if line.startswith(f"{i+1}.")), None)
        if match_line is None:
            raise ValueError(
                f"LLM did not return an answer for post {i+1}.\n"
                f"LLM output:\n{response.output_text}"
            )
        #Extract True/False
        answer_text = match_line.split(".", 1)[1].strip().lower()
        results.append(answer_text == "true")  # just True/False

    return results

#Prompt 2:

def chatgpt_prompt_2(posts_batch):
    posts_text = "\n".join(f"{i+1}. {post}" for i, post in enumerate(posts_batch))
    prompt = f"""
        You will be given Turkish social media posts. Go through the entire post, word by word, and detect
        if it contains any English words mixed into the Turkish, including individual English words
        or sentences/phrases in English. Don't count URLs or brand names as English words.

        If English words are present in the post, return true.

        Steps 
        1. Language Detection: 
        Detect whether the given post 
        contains English words. 

        Output Format
        Detection Result: Output "True" if English words are present, 
        otherwise "False".
        Number your answers to match the post numbers, and put each response on its own line.

        Example output:
        1. True
        2. False
        etc

        Posts:
        {posts_text}
    """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )

    #Parse numbered output lines
    lines = [line.strip() for line in response.output_text.strip().split("\n") if line.strip()]

    results = []
    for i, post in enumerate(posts_batch):
        #Find the corresponding line starting with the post number
        match_line = next((line for line in lines if line.startswith(f"{i+1}.")), None)
        if match_line is None:
            raise ValueError(
                f"LLM did not return an answer for post {i+1}.\n"
                f"LLM output:\n{response.output_text}"
            )
        #Extract True/False
        answer_text = match_line.split(".", 1)[1].strip().lower()
        results.append(answer_text == "true")  #just True/False

    return results


def run_prompts_in_batches(df, batch_size=50):
    prompt_1_results = []
    prompt_2_results = []

    entries = df['entry'].tolist()
    total = len(entries)

    for i in range(0, total, batch_size):
        batch = entries[i:i+batch_size]

        print(f"Processing posts {i+1} to {i+len(batch)}...")

        #Run both prompts on this batch
        batch_prompt_1 = chatgpt_prompt_1(batch)
        batch_prompt_2 = chatgpt_prompt_2(batch)

        #Append results
        prompt_1_results.extend(batch_prompt_1)
        prompt_2_results.extend(batch_prompt_2)

    #Add results to DataFrame
    df['prompt_1'] = prompt_1_results
    df['prompt_2'] = prompt_2_results

    return df


if __name__ == "__main__":
    asyncio.run(fetch_and_save_entries(topic_names))

    all_entries = pd.read_csv("eksi_entries_200pgs.csv")

    #Run the filter
    filtered_by_langdetect = filter_code_mixed_entries_df(
        all_entries,
        is_code_mixed_langdetect,
        output_file="eksi_entries_200pgs_filtered.csv"
    )

    all_filtered = pd.read_csv("eksi_entries_200pgs_filtered.csv")
    all_filtered['word_count'] = all_filtered['entry'].apply(lambda x: len(x.split()))
    client=OpenAI(api_key="key")
    all_filtered = run_prompts_in_batches(all_filtered, batch_size=50)

    #To CSV
    all_filtered.to_csv("eksi_entries_200pgs_GPT.csv")



