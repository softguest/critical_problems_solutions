import { auth,} from "@/auth";
import { UserCircle } from "lucide-react";
import Link from "next/link";
import SignOut from "./SignOut";
import SearchBar from "./SearchBar";

export default async function Navbar() {

  const session = await auth();

  return (
    <header className="sticky top-0 z-10 w-full bg-white shadow-sm px-4 py-3 flex items-center justify-between md:pl-4">
      <div className="flex items-center space-x-2">
        <Link href="/dashboard">
          <h1 className="text-1xl md:text-3xl md:mr-10 font-semibold">CP<span className="text-blue-700 font-bold">S</span></h1>
        </Link>
        <div className="hidden md:block">
          <SearchBar />
        </div>
      </div>
      <div className="space-x-2 flex items-center">
        {/* Placeholder for user profile, notifications etc. */}
        <span className="text-[12px] md:text-sm text-gray-600 md:ml-16">
          Hi {session?.user?.firstName}
        </span>
        <div className="flex items-center space-x-2">
          <Link href="/profile">
            <UserCircle size={30} />
          </Link>
          <SignOut />
        </div>
      </div>
    </header>
  )
}
