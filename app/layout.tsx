import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { SessionProvider } from 'next-auth/react'
import { auth } from '@/auth'
import Providers from '@/components/Providers'
import './globals.css'
import { Toaster } from '@/components/ui/Toaster'
import { cn } from '@/lib/utils'

import { Poppins } from "next/font/google";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "600"],
  variable: "--font-poppins",
});

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: "Critical Problems and Solutions",
  description: "AI-powered business solutions platform",
}

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const session = await auth();

  return (
    <SessionProvider session={session}>
      <html lang="en">
        <body 
        // className="min-h-[80vh]"
        >
          <Providers>
            {children}
          </Providers>
        </body>
      </html>
    </SessionProvider>
  )
}
