export default function RoadmapView({ roadmap }) {
    if (!roadmap) return <p>No roadmap selected.</p>;
  
    return (
      <div className="roadmap-view">
        {Object.entries(roadmap).map(([week, days]) => (
          <div key={week}>
            <h2>{week}</h2>
            {Object.entries(days).map(([day, tasks]) => (
              <div key={day}>
                <h3>{day}</h3>
                <ul>
                  {tasks.map((task, i) => (
                    <li key={i}>
                      <strong>{task.topic}</strong> — <a href={task.resource} target="_blank" rel="noreferrer">{task.resource}</a> ({task.hours} hrs)
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        ))}
      </div>
    );
  }
  