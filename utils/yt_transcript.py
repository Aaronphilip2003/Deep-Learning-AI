from youtube_transcript_api import YouTubeTranscriptApi

video_id = ''  # e.g., 'dQw4w9WgXcQ'
transcript = YouTubeTranscriptApi.get_transcript(video_id)

with open ("./transcript.txt",'w+') as f:
    for entry in transcript:
        f.write(entry['text'])
