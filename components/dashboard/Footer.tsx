export default function Footer() {
  return (
    <footer className="w-full bottom-0 text-center py-4 text-sm text-gray-500 border-t md:pl-64">
      © {new Date().getFullYear()} CPS. All rights reserved.
      {/* © {new Date().getFullYear()} C<span className="hidden md:block">ritical </span>P<span className="hidden md:block">roblems & </span>S<span className="hidden md:block">olutions</span>. All rights reserved. */}
    </footer>
  )
}
