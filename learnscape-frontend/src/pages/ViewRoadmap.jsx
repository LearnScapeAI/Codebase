import { useParams } from "react-router-dom";
import RoadmapView from "../components/RoadmapView";

export default function ViewRoadmap({ roadmaps }) {
  const { id } = useParams();
  const roadmap = roadmaps[parseInt(id)];

  return (
    <div>
      <h2>📘 Roadmap Details</h2>
      <RoadmapView roadmap={roadmap?.roadmap} />
    </div>
  );
}
