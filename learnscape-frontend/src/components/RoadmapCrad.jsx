import { Link } from "react-router-dom";

export default function RoadmapCard({ roadmap, index }) {
  return (
    <div className="roadmap-card">
      <h3>Roadmap #{index + 1}</h3>
      <p><strong>Skill:</strong> {roadmap.skill}</p>
      <Link to={`/view/${index}`}>
        <button>View Details</button>
      </Link>
    </div>
  );
}
