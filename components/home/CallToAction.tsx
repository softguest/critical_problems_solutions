// === components/CallToAction.tsx ===
export default function CallToAction() {
  return (
    <section className="w-full max-w-4xl text-center py-12">
      <h2 className="text-3xl font-bold mb-4">Ready to solve your business challenges?</h2>
      <p className="text-lg text-gray-600 mb-6">Join thousands of businesses using AI to unlock growth and efficiency.</p>
      <button className="px-6 py-3 rounded-lg bg-green-600 text-white hover:bg-green-700 transition">
        Get Started Free
      </button>
    </section>
  );
}