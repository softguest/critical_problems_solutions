import { ExtendedUser } from "@/next-auth";
import { 
  Card, 
  CardContent, 
  CardHeader
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface UserInfoProps {
  user?: ExtendedUser;
  label: string;
};

export const UserInfo = ({
  user,
  label,
}: UserInfoProps) => {
  return (
        <div>
            <div className="grid xl:grid-cols-2 gap-4">
                {/* PERSONAL INFORMATION DETAILS  */}
                <div className="p-2 rounded-md bg-slate-50 space-y-2">
                  <span className="text-2xl text-slate-500">Personal Details:</span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white mt-2">
                      <p className="text-sm font-medium"> 
                        Names 
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {user?.firstName} {user?.lastName}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                        Email
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {user?.email}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                        Role
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {user?.role}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                        Date Of Birth
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {user?.dateOfBirth}
                      </p>
                    </div>
                </div>

                {/* CONTACT SECTION  */}
                <div className="p-2 rounded-md bg-slate-50 space-y-2">
                  <span className="text-2xl text-slate-500">Address:</span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white mt-2">
                      <p className="text-sm font-medium">
                        Country Code
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.countryCode} */}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                      Country Of Residence
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.countryofResidence} */}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                          Postal Address
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.postalAddress} */}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                          Residential Address
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.residentialAddress} */}
                      </p>
                    </div>
                </div>

                {/* SPONSOR INFORMATION  */}
                <div className="p-2 rounded-md bg-slate-50 space-y-2">
                  <span className="text-2xl text-slate-500">Sponsors Details:</span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white mt-2">
                      <p className="text-sm font-medium">
                          Name Of Sponsor
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.name} */}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                          Address Of Sponsor
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.residentialAddress} */}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                          Ocupation Of Sponsor
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {/* {user?.postalAddress} */}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 rounded-lg border p-2 shadow-sm bg-white">
                      <p className="text-sm font-medium">
                          Telephone Of Sponsor
                      </p>
                      <p className="truncate font-semibold text-left sm:text-right">
                        {user?.telePhone}
                      </p>
                    </div>
                </div>
            </div>

            <div className="flex flex-row items-center justify-between rounded-lg border p-2 md:p-4 shadow-sm mt-4">
              <p className="text-sm font-medium">
                Two Factor Authentication Status
              </p>
              <Badge 
                variant={user?.isTwoFactorEnabled ? "success" : "destructive"}
              >
                {user?.isTwoFactorEnabled ? "ON" : "OFF"}
              </Badge>
            </div>
        </div>
  )
}