import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { generateRoadmapAPI } from "../utils/api";

export default function GenerateRoadmap({ onAddRoadmap }) {
  const [skill, setSkill] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleGenerate = async () => {
    setLoading(true);
    const roadmapData = await generateRoadmapAPI(skill);
    if (roadmapData) {
      onAddRoadmap({ skill, roadmap: roadmapData });
      navigate("/");
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>Generate a Roadmap</h2>
      <input
        type="text"
        placeholder="e.g. R, SQL, React"
        value={skill}
        onChange={(e) => setSkill(e.target.value)}
      />
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Generating..." : "Generate"}
      </button>
    </div>
  );
}
