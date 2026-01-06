async function predict() {
  const res = await fetch("http://127.0.0.1:8000/predict-yield?rainfall=120&temperature=28&fertilizer=50",
    { method: "POST" });
  const data = await res.json();
  document.getElementById("result").innerText =
    "Predicted Yield: " + data.predicted_yield;
}
