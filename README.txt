URDU LECTURE TRANSCRIPTION (Naima Sohaib)
=========================================

Turns Urdu lecture recordings into Word documents (.docx), right-to-left, in
paragraphs with timestamps. Everything runs on your own PC; audio is never uploaded.


ONE-TIME SETUP
--------------
1. Install Python 3.10 or newer from https://www.python.org/downloads/
   During setup, tick "Add python.exe to PATH".

2. Install FFmpeg. Open PowerShell and run:
       winget install Gyan.FFmpeg
   Then close and reopen PowerShell.

3. Put all the files from this folder together in one folder, for example
   C:\Transcription


TRANSCRIBING
------------
Open PowerShell in that folder (in File Explorer: click the address bar,
type powershell, press Enter), then run:

   One lecture:
       powershell -ExecutionPolicy Bypass -File .\Transcribe.ps1 "C:\Lectures\lecture1.mp3"

   A whole folder of lectures:
       powershell -ExecutionPolicy Bypass -File .\Transcribe.ps1 "C:\Lectures"

The Word documents appear in the "transcripts" folder next to the script.

The first run takes a few extra minutes: it installs the Python libraries and
downloads the speech model (about 1.6 GB). Later runs start straight away.


HOW LONG IT TAKES
-----------------
About 50 minutes per hour of audio on a typical laptop. A folder of lectures is
best left running overnight.

Progress is saved about once a minute. If the run stops for any reason (Ctrl+C,
the PC sleeps or restarts, power cut), run the same command again: finished
lectures are skipped and an unfinished one continues from where it stopped.
(While running, a <name>.partial.json file in the transcripts folder holds the
progress; it is removed when the lecture is done.)


FIXING RECURRING MISTAKES
-------------------------
If the same word is misspelled again and again, add it to corrections.tsv:
    wrong word<TAB>correct word
(one per line, separated by a Tab). Then rebuild the documents without
transcribing again (takes seconds):
       powershell -ExecutionPolicy Bypass -File .\Transcribe.ps1 "C:\Lectures" --rebuild

Names and terms the speaker uses often can also be added to prompt.txt, which
guides the model's spelling for new transcriptions.

corrections.learned.tsv holds fixes learned from verified lectures. Do not edit
it by hand; it is replaced when the training is updated.


USEFUL OPTIONS (add after the file or folder)
-------------------------------------------
    --start 00:10:00 --duration 00:05:00   transcribe only part, to test quickly
    --no-timestamps                        leave timestamps out of the document
    --overwrite                            transcribe again even if already done
    --help                                 list every option


NOTES
-----
* Accuracy: about 9 in 10 words match a verified transcript. Always proofread
  before publishing; the document header says "machine transcript, please verify".
* Quran verses and Arabic duas are detected and written in Arabic script
  (shown centred, in blue); recited Arabic may still contain small errors.
* Supported audio: mp3, wav, m4a, aac, ogg, opus, flac, wma, and the audio of
  mp4, mkv and webm videos.


TROUBLESHOOTING
---------------
"Python was not found"   Install Python (step 1) and tick "Add python.exe to PATH".
"FFmpeg was not found"   Run step 2, then close and reopen PowerShell.
Very slow                 Close other heavy programs; it uses all processor cores.
