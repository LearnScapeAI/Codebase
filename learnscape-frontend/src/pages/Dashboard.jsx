import RoadmapCard from "../components/RoadmapCard";
import { useNavigate } from "react-router-dom";

export default function Dashboard({ user, roadmaps }) {
  const navigate = useNavigate();

  return (
    <div>
      <h1>Welcome, {user} 👋</h1>
      <button onClick={() => navigate("/generate")}>+ New Roadmap</button>

      <div className="roadmap-list">
        {roadmaps.length > 0 ? (
          roadmaps.map((rm, i) => (
            <RoadmapCard key={i} roadmap={rm} index={i} />
          ))
        ) : (
          <p>No roadmaps yet.</p>
        )}
      </div>
    </div>
  );
}
