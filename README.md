# Happy Bhai Dooj Streamlit App 🪔

A warm, playful Bhai Dooj greeting app with personalized names, festive decorations, five ML-inspired wishes, and a fully synthetic mini data lab.

## Run locally

1. Use Python 3.10 or newer.
2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and add `app.py`, `requirements.txt`, and this README at its top level.
2. Commit and push the files to GitHub.
3. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Click **Deploy**. Streamlit installs the packages listed in `requirements.txt` automatically.

## What the ML section means

The dataset is generated from a chosen seed and number of pretend memories. It includes simulated playful fights, shared laughs, care, and love scores, with a live correlation graph for fights, care, and love. It is a visual, educational example of correlation, linear regression, and logistic classification—not evidence of a causal relationship in real sibling bonds.

## Customizing it

Change the default names in `app.py`, edit the messages in the `MESSAGES` list, or adjust colours in `apply_style()` to make the celebration your own.
