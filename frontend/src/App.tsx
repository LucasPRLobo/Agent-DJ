import { BrowserRouter, Routes, Route } from "react-router-dom";
import { HostView } from "./pages/HostView";
import { GuestView } from "./pages/GuestView";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HostView />} />
        <Route path="/guest/:sessionId" element={<GuestView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
