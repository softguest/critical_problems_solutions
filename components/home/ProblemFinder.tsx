// === components/ProblemFinder.tsx ===
export default function ProblemFinder() {
  return (
    <section className="w-full max-w-4xl">
      <h2 className="text-2xl font-semibold mb-4">Or select a problem category</h2>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {["Marketing", "Sales", "Operations", "HR", "Finance", "Customer Support"].map((category) => (
          <button
            key={category}
            className="px-4 py-3 bg-white border border-gray-300 rounded-lg shadow-sm hover:bg-blue-50"
          >
            {category}
          </button>
        ))}
      </div>
    </section>
  );
}