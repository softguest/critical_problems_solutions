'use client'

import { useState } from 'react'
import { Menu, X } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { FaBookOpen, FaCog, FaCreativeCommons, FaHome, FaRobot, FaUserCircle, FaUserMd } from 'react-icons/fa'
import SearchBar from './SearchBar'

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false)

  const toggleSidebar = () => setIsOpen(!isOpen)

  return (
    <>
      {/* Mobile Toggle Button */}
      <div className="md:hidden p-4">
        <button onClick={toggleSidebar}>
          {isOpen ? <X className="h-6 w-6 text-cyan-800" /> : <Menu className="h-6 w-6 text-cyan-800" />}
        </button>
      </div>

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-30 w-60 bg-white shadow-md transform transition-transform duration-300 ease-in-out md:translate-x-0 md:static md:block',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >

      <nav className="flex flex-col h-full p-4 space-y-6">
        <h2 className="text-xl text font-bold mb-4">Dashboard</h2>
        <div className='md:hidden'>
          <SearchBar />
        </div>
        <Link href="/" className="flex items-center space-x-2 hover:text-indigo-600">
          <FaHome />
          <span>Home</span>
        </Link>

        <Link href="/admin/problem" className="flex items-center space-x-2 hover:text-indigo-600">
          <FaUserCircle />
          <span>Problems</span>
        </Link>

        <Link href="/admin/solution" className="flex items-center space-x-2 hover:text-indigo-600">
          <FaUserMd />
          <span>Solutions</span>
        </Link>

        {/* <Link href="/article" className="flex items-center space-x-2 hover:text-indigo-600">
          <FaBookOpen />
          <span>Articles</span>
        </Link> */}

        <Link href="/dashboard/settings" className="flex items-center space-x-2 text-indigo-600 hover:text-indigo-600">
          <FaCog />
          <span>Settings</span>
        </Link>
        {/* <Link href="/dashboard/articles/create" className="flex items-center space-x-2 hover:text-indigo-600">
          <FaCreativeCommons />
          <span>Create Article</span>
        </Link> */}
      </nav>
      </aside>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          onClick={toggleSidebar}
          className="fixed inset-0 bg-black/50 z-20 md:hidden"
        />
      )}
    </>
  )
}
