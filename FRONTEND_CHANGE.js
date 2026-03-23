// ─────────────────────────────────────────────────────────────────
// WHAT TO CHANGE IN YOUR TakeExamPage.tsx
// Only ONE small addition needed — send token + session_id to Python
// ─────────────────────────────────────────────────────────────────

// FIND this section in startHeadDetection():
//
//   ws.onopen = () => setWsStatus("Connected");
//
// REPLACE it with this:

ws.onopen = () => {
  setWsStatus("Connected");

  // Send token + session_id to Python so it can save to MongoDB
  const token = localStorage.getItem("token");
  ws.send(JSON.stringify({
    type:       "auth",
    token:      token,
    session_id: sessionIdRef.current,
  }));
};

// ─────────────────────────────────────────────────────────────────
// That's it! No other changes needed in your frontend.
// Your frontend already:
//   ✅ Connects to ws://localhost:8000/ws
//   ✅ Sends JPEG frames every 200ms
//   ✅ Receives pose results and updates UI
//   ✅ Counts head movements and detects cheating
//   ✅ Logs tab switches
// ─────────────────────────────────────────────────────────────────
