import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState } from "react";
import Auth from "./components/Auth";
import Dashboard from "./pages/Dashboard";
import GenerateRoadmap from "./pages/GenerateRoadmap";
import ViewRoadmap from "./pages/ViewRoadmap";
import "./index.css";

export default function App() {
  const [user, setUser] = useState(null);
  const [roadmaps, setRoadmaps] = useState([]);

  const addRoadmap = (newRM) => {
    setRoadmaps((prev) => [...prev, newRM]);
  };

  if (!user) return <Auth onLogin={setUser} />;

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard user={user} roadmaps={roadmaps} />} />
        <Route path="/generate" element={<GenerateRoadmap onAddRoadmap={addRoadmap} />} />
        <Route path="/view/:id" element={<ViewRoadmap roadmaps={roadmaps} />} />
      </Routes>
    </Router>
  );
}
