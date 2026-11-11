"""A festive, illustrative Streamlit app for Bhai Dooj."""

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Happy Bhai Dooj ✨",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="collapsed",
)


MESSAGES = [
    '"Our sibling bond has no overfitting, no underfitting, and absolutely zero chance of data leakage. ❤️"',
    '"You are my favourite feature, and our bond has the highest possible correlation. 😂"',
    '"May your life have maximum accuracy, minimum loss, and infinite happiness. Happy Bhai Dooj!"',
    '"No matter how many epochs life runs, my affection for you will never converge to zero."',
    '"Among billions of data points in this universe, you are my most precious training sample. Happy Bhai Dooj!"',
]


def apply_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=DM+Sans:wght@400;500;700&display=swap');
        .stApp { background: radial-gradient(circle at 8% 8%, #fff1be 0, transparent 22%),
                 radial-gradient(circle at 92% 12%, #ffd5e2 0, transparent 24%), #fffaf0; color: #4b2635; }
        h1, h2, h3 { font-family: 'Baloo 2', sans-serif !important; color: #8e294b; }
        .hero { position: relative; overflow: hidden; padding: 2.5rem 2rem 2rem; border-radius: 28px;
                 background: linear-gradient(125deg, #8b2448, #c24460 55%, #f08b62); color: white;
                 box-shadow: 0 16px 36px rgba(130, 38, 70, .22); text-align: center; }
        .hero h1 { color: #fff8d2; font-size: clamp(2.3rem, 6vw, 4.5rem); line-height: 1; margin: .2rem 0 .5rem; }
        .hero p { font-family: 'DM Sans', sans-serif; font-size: 1.12rem; margin: 0; }
        .garland { color: #ffde73; font-size: 1.45rem; letter-spacing: .34rem; white-space: nowrap; }
        .float { position: absolute; font-size: 2rem; animation: bob 3s ease-in-out infinite; }
        .f1 { top: 18px; left: 7%; } .f2 { top: 25px; right: 8%; animation-delay: .7s; }
        .f3 { bottom: 4px; left: 20%; animation-delay: 1.1s; }
        @keyframes bob { 50% { transform: translateY(-8px) rotate(5deg); } }
        .message-card { min-height: 136px; border: 1px solid #f4d4a4; border-radius: 18px; padding: 1.1rem;
                         background: linear-gradient(145deg, #fffef9, #fff4df); box-shadow: 0 5px 16px rgba(129,65,42,.09); }
        .message-card p { font-family: 'Baloo 2', sans-serif; font-size: 1.06rem; line-height: 1.35; color: #6d3141; }
        .diya-scene { min-height: 245px; display: flex; align-items: center; justify-content: center; flex-direction: column;
                      border-radius: 22px; background: linear-gradient(160deg, #4d1935, #9e3152 55%, #df7d55);
                      box-shadow: inset 0 0 45px rgba(255, 218, 113, .2); color: #fff7d6; text-align: center; }
        .diya-scene .flame { font-size: 5.7rem; line-height: 1; filter: drop-shadow(0 0 18px #ffde6c); animation: glow 2s ease-in-out infinite; }
        .diya-scene p { margin: .35rem 1rem 0; font-family: 'Baloo 2', sans-serif; font-size: 1.25rem; }
        @keyframes glow { 50% { transform: scale(1.08); filter: drop-shadow(0 0 28px #fff5a6); } }
        .metric-card { background: #fff; border-left: 5px solid #e6754e; border-radius: 12px; padding: 1rem; }
        .note { background: #fff3c9; border: 1px dashed #d58b39; border-radius: 14px; padding: 1rem 1.2rem; }
        div[data-testid="stMetric"] { background: #fff; border-radius: 14px; padding: .7rem 1rem; box-shadow: 0 4px 12px rgba(112,58,49,.08); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def make_memories(seed: int, count: int) -> pd.DataFrame:
    """Create a deliberately synthetic, positively associated example dataset."""
    rng = np.random.default_rng(seed)
    fights = rng.integers(1, 13, count)
    shared_laughs = np.clip(fights * 1.55 + rng.normal(5.5, 3.2, count), 0, 28)
    care = np.clip(42 + fights * 3.3 + shared_laughs * 1.25 + rng.normal(0, 8, count), 30, 100)
    love = np.clip(38 + care * 0.38 + shared_laughs * 0.45 + rng.normal(0, 8, count), 30, 100)
    return pd.DataFrame(
        {
            "Memory": [f"Memory {i + 1}" for i in range(count)],
            "Playful fights this month": fights,
            "Shared laughs": shared_laughs.round(0).astype(int),
            "Care & affection score": care.round(1),
            "Love & affection score": love.round(1),
        }
    )


def correlation_graph(data: pd.DataFrame) -> alt.LayerChart:
    """Draw pairwise correlations among playful fights, care, and love."""
    variables = [
        ("Playful fights this month", "Fight", 0.0, 0.0, "#e6754e"),
        ("Care & affection score", "Care", 1.0, 0.0, "#2c8a83"),
        ("Love & affection score", "Love", 0.5, 0.86, "#c24460"),
    ]
    coordinates = {column: (x, y) for column, _, x, y, _ in variables}
    pairs = [
        ("Playful fights this month", "Care & affection score", 0.5, -0.1),
        ("Care & affection score", "Love & affection score", 0.79, 0.5),
        ("Love & affection score", "Playful fights this month", 0.21, 0.5),
    ]
    edge_points = []
    edge_labels = []
    for left, right, label_x, label_y in pairs:
        correlation = float(data[left].corr(data[right]))
        pair = f"{left} ↔ {right}"
        for column in (left, right):
            x, y = coordinates[column]
            edge_points.append(
                {
                    "pair": pair,
                    "x": x,
                    "y": y,
                    "correlation": correlation,
                    "strength": abs(correlation),
                }
            )
        edge_labels.append(
            {"x": label_x, "y": label_y, "label": f"r = {correlation:+.2f}"}
        )

    nodes = [
        {"x": x, "y": y, "name": name, "color": color}
        for _, name, x, y, color in variables
    ]
    x_scale = alt.Scale(domain=[-0.2, 1.2])
    y_scale = alt.Scale(domain=[-0.22, 1.08])
    edges = (
        alt.Chart(pd.DataFrame(edge_points))
        .mark_line(opacity=0.8)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            detail="pair:N",
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                legend=alt.Legend(title="Correlation (r)"),
            ),
            size=alt.Size("strength:Q", scale=alt.Scale(range=[2, 8]), legend=None),
            tooltip=[
                alt.Tooltip("pair:N", title="Pair"),
                alt.Tooltip("correlation:Q", title="Correlation (r)", format=".2f"),
            ],
        )
    )
    edge_values = (
        alt.Chart(pd.DataFrame(edge_labels))
        .mark_text(fontSize=13, fontWeight="bold", color="#6d3141")
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            text="label:N",
        )
    )
    node_circles = (
        alt.Chart(pd.DataFrame(nodes))
        .mark_circle(size=1900, stroke="white", strokeWidth=3)
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip="name:N",
        )
    )
    node_names = (
        alt.Chart(pd.DataFrame(nodes))
        .mark_text(color="white", fontSize=15, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            text="name:N",
        )
    )
    return (
        alt.layer(edges, edge_values, node_circles, node_names)
        .properties(height=420)
        .configure_view(stroke=None)
    )


def main() -> None:
    apply_style()
    st.markdown(
        """<section class="hero">
        <span class="float f1">🌼</span><span class="float f2">🪔</span><span class="float f3">🌸</span>
        <div class="garland">✿ ✦ ✿ ✦ ✿ ✦ ✿</div>
        <h1>Happy Bhai Dooj</h1>
        <p>A tiny celebration of teasing, caring, and the beautiful bond that keeps growing.</p>
        <div class="garland">🪔 🌼 🪔 🌼 🪔</div>
        </section>""",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Make it yours 💌")
        sibling_one = st.text_input("Your name", value="Sharvin")
        sibling_two = st.text_input("Sibling's name", value="Sharvi")
        safe_sibling_one = html.escape(sibling_one)
        safe_sibling_two = html.escape(sibling_two)
        st.markdown(
            f"""<div class="message-card"><p>Dear {safe_sibling_two},<br><br>
            From every silly argument to every quiet act of care, you make life brighter, funnier, and more meaningful.
            Here’s to our unbreakable bond—Happy Bhai Dooj!<br><br>With love, {safe_sibling_one} 🌼</p></div>""",
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("A festive note 🪔")
        st.markdown(
            '<div class="diya-scene"><div>🌼 &nbsp; ✨ &nbsp; 🌼</div><div class="flame">🪔</div>'
            '<p>May this diya light up every shared memory.</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("ML-flavoured wishes ✨")
    message_columns = st.columns(3)
    for index, message in enumerate(MESSAGES):
        with message_columns[index % 3]:
            st.markdown(f'<div class="message-card"><p>{message}</p></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("The sibling-memory lab 📊")
    st.caption("Explore a friendly synthetic dataset—not a study of real siblings.")
    with st.container(border=True):
        st.subheader("Tune the simulation 🎛️")
        st.caption(
            "Choose how many pretend memories to mix and a seed to replay the same "
            "simulation."
        )
        seed_column, count_column = st.columns(2)
        with seed_column:
            seed = st.slider("Memory-mixer seed", 1, 100, 22)
        with count_column:
            count = st.slider("Number of pretend memories", 20, 120, 50, step=5)
    data = make_memories(seed, count)
    x = data[["Playful fights this month"]]
    y = data["Care & affection score"]
    regressor = LinearRegression().fit(x, y)
    correlation = data["Playful fights this month"].corr(y)
    r_squared = r2_score(y, regressor.predict(x))
    a, b, c = st.columns(3)
    a.metric("Fight ↔ care correlation (r)", f"{correlation:.2f}")
    b.metric("Regression R²", f"{r_squared:.2f}")
    b.caption("In this synthetic sample")
    c.metric("Fight ↔ care trend", "Positive" if correlation >= 0 else "Negative")

    st.markdown("#### How the three signals move together")
    st.caption(
        "Each edge shows a Pearson correlation from this simulated sample. "
        "Thicker edges mean a stronger association, not cause and effect."
    )
    st.altair_chart(correlation_graph(data), use_container_width=True)

    st.markdown("#### One relationship, memory by memory")
    chart_data = data.copy()
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=95, color="#c24460", opacity=0.75)
        .encode(
            x=alt.X("Playful fights this month:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Care & affection score:Q", scale=alt.Scale(domain=[25, 105])),
            tooltip=["Memory:N", "Playful fights this month:Q", "Shared laughs:Q", "Care & affection score:Q"],
        )
        .properties(height=380)
    )
    trend = (
        alt.Chart(chart_data)
        .transform_regression("Playful fights this month", "Care & affection score")
        .mark_line(color="#f3a33b", strokeWidth=4)
        .encode(x="Playful fights this month:Q", y="Care & affection score:Q")
    )
    st.altair_chart(chart + trend, use_container_width=True)
    st.markdown(
        """<div class="note"><b>Kind data reminder:</b> This playful dataset is generated by this app to demonstrate a scatter plot,
        correlation, and regression. It illustrates an association built into the simulation; it does <b>not</b> show that quarrels cause love or care.
        Real relationships are wonderfully more complex than a model. 💛</div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Can the model spot a high-care memory? 🤖")
    features = data[["Playful fights this month", "Shared laughs"]]
    target = (data["Care & affection score"] >= data["Care & affection score"].median()).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=seed, stratify=target)
    classifier = LogisticRegression().fit(x_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    model_col, try_col = st.columns([1, 1.5])
    with model_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Demo classifier accuracy", f"{accuracy:.0%}")
        st.caption("Logistic regression on a held-out slice of this synthetic data.")
        st.markdown("</div>", unsafe_allow_html=True)
    with try_col:
        proposed_fights = st.slider("Playful fights in a new pretend month", 0, 15, 6)
        proposed_laughs = st.slider("Shared laughs in that month", 0, 30, 15)
        probability = classifier.predict_proba(pd.DataFrame([[proposed_fights, proposed_laughs]], columns=features.columns))[0, 1]
        st.info(f"This illustrative model estimates a **{probability:.0%}** chance of a high-care memory. The best real-world feature is kindness. 💗")

    st.markdown("<p style='text-align:center; font-size:1.15rem; margin-top:2.5rem;'>🌸 May your bond keep blooming through every laugh, lesson, and loving little squabble. 🌸</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
