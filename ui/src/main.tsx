import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

const bootSplash = document.getElementById("app-boot-splash");
if (bootSplash) {
  requestAnimationFrame(() => {
    bootSplash.classList.add("is-hidden");
    window.setTimeout(() => bootSplash.remove(), 220);
  });
}
