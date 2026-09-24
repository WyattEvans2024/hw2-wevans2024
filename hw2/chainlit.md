# Study Notes Assistant

FAU ID: **wevans2024**

Upload TXT or JSON study notes, then ask a question about them. Upload up to
5 files at a time, with a maximum size of 10 MB per file.

- Use **/upload** to open the upload prompt, or attach files to your message.
- Use **/documents** to see the available filenames and note titles.
- Answers include the retrieved source filenames, titles, and short excerpts.
- If your notes do not contain the answer, the assistant will say so.

JSON files use this format:

```json
{"notes": [{"title": "Note title", "content": "Note text"}]}
```

Try the demo notes and ask: **What does chunk overlap help preserve?**
