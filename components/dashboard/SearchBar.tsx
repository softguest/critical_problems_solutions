'use client'

import { Prisma, User } from '@prisma/client'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import debounce from 'lodash.debounce'
import { usePathname, useRouter } from 'next/navigation'
import { FC, useCallback, useEffect, useRef, useState } from 'react'

import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/Command'

import { useOnClickOutside } from '@/hooks/use-on-click-outside'
import { Users } from 'lucide-react'
import { FaBuilding } from 'react-icons/fa6'
import { FaUser } from 'react-icons/fa'
import Image from 'next/image'

interface SearchBarProps {}

const SearchBar: FC<SearchBarProps> = ({}) => {
  const [input, setInput] = useState<string>('')
  const pathname = usePathname()
  const commandRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  useOnClickOutside(commandRef, () => {
    setInput('')
  })

  const request = debounce(async () => {
    refetch()
  }, 300)

  const debounceRequest = useCallback(() => {
    request()

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const {
    isFetching,
    data: queryResults,
    refetch,
    isFetched,
  } = useQuery({
    queryFn: async () => {
      if (!input) return []
      const { data } = await axios.get(`/api/search?q=${input}`)
      return data as (User & {
        _count: Prisma.UserCountOutputType
      })[]
    },
    queryKey: ['search-query'],
    enabled: false,
  })

  useEffect(() => {
    setInput('')
  }, [pathname])

  return (
    <Command
      ref={commandRef}
      className='relative rounded-lg border max-w-lg z-50 overflow-visible'>
      <CommandInput
        isLoading={isFetching}
        onValueChange={(text) => {
          setInput(text)
          debounceRequest()
        }}
        value={input.toLowerCase()}
        className='outline-none border-none focus:border-none focus:outline-none ring-0'
        placeholder='Search Profile...'

      />

      {input.length > 0 && (
        <CommandList className='absolute bg-white top-full inset-x-0 shadow rounded-b-md'>
          {isFetched && <CommandEmpty>No results found.</CommandEmpty>}
          {(queryResults?.length ?? 0) > 0 ? (
            <CommandGroup heading='Users'>
              {queryResults?.map((user) => (
                <CommandItem
                  onSelect={(e) => {
                    router.push(`/profile/${e}`)
                    router.refresh()
                  }}
                  key={user.id}
                  value={user.email as string || user.firstName as string || user.lastName as string || user.bio as string || user.profession as string}>
                  {/* <Image src={user.profileImage ?? ""} alt="user's Image" width={30} height={30} className="mr-4 rounded-full" /> */}
                    {user?.profileImage ? (
                      <Image
                        src={user.profileImage}
                        alt="User"
                        width={30}
                        height={30}
                        className="rounded-full object-cover mr-2"
                      />
                    ) : (
                      <div className="w-3 h-4 bg-slate-500 p-2 text-white flex items-center justify-center rounded-full font-bold mr-2">
                        {user?.firstName
                          ?.split(" ")
                          ?.slice(0, 1)
                          ?.map((word: string) => word.charAt(0).toUpperCase())
                          ?.join("")}
                      </div>
                    )}
                  <a href={`/profile/${user.id}`} className=""><span className="text-slate-500">{user.cityName} {user.profession}</span> ... <span className="text-[#faa635]">{user.role}</span></a>
                </CommandItem>
              ))}
            </CommandGroup>
          ) : null}
        </CommandList>
      )}
    </Command>
  )
}

export default SearchBar
