// === components/SolutionCards.tsx ===
export default function SolutionCards() {
  return (
    <section className="w-full max-w-6xl">
      <h2 className="text-2xl font-semibold mb-6">Recommended Solutions</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3].map((id) => (
          <div
            key={id}
            className="bg-white border border-gray-200 rounded-xl p-6 shadow-md hover:shadow-lg transition"
          >
            <h3 className="text-xl font-bold mb-2">Solution Title {id}</h3>
            <p className="text-gray-600 mb-4">
              Brief description of the solution. Explains how it addresses the problem.
            </p>
            <button className="text-blue-600 hover:underline">Learn More</button>
          </div>
        ))}
      </div>
    </section>
  );
}