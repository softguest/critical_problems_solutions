// === components/HeroSection.tsx ===
export default function HeroSection() {
  return (
    <section className="w-full max-w-5xl text-center">
      <h1 className="text-4xl md:text-6xl font-bold mb-4">
        Solve Business Problems with AI-Powered Recommendations
      </h1>
      <p className="text-lg md:text-xl text-gray-600 mb-8">
        Describe your problem and get tailored solutions instantly.
      </p>
      <div className="flex flex-col md:flex-row gap-4 justify-center">
        <input
          type="text"
          placeholder="e.g. Low customer retention..."
          className="w-full md:w-2/3 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button className="px-6 py-3 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition">
          Find Solutions
        </button>
      </div>
    </section>
  );
}