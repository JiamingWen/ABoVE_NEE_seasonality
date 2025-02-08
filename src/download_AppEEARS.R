library(httr)
library(jsonlite)
library(curl)

year = 2014 # 2012 2013 2014 2017

# Create a token by calling AppEEARS API login service. Update the “USERNAME” and “PASSEWORD” with yours below
secret <- base64_enc(paste("jw2495", "Baobao950404", sep = ":"))
response <- POST("https://appeears.earthdatacloud.nasa.gov/api/login",
                 add_headers("Authorization" = paste("Basic", gsub("\n", "", secret)),
                             "Content-Type" = "application/x-www-form-urlencoded;charset=UTF-8"),
                 body = "grant_type=client_credentials")
token_response <- prettify(toJSON(content(response), auto_unbox = TRUE))

# Create a handle
s = new_handle()
handle_setheaders(s, 'Authorization'=paste("Bearer", fromJSON(token_response)$token))

# Loop through the URLs and downlaod outputs
filesl = read.table(paste0('/central/groups/carnegie_poc/jwen2/ABoVE/modis_fpar/data_',year,'/Alaskafpar',year,'-download-list.txt'))[,1]
dataPath = paste0('/central/groups/carnegie_poc/jwen2/ABoVE/modis_fpar/data_',year)
for (d in 1:length(filesl)){
  curl_download(url=as.character(filesl[d]), destfile=paste0(dataPath,"/", basename(as.character(filesl[d]))), handle = s)
  Sys.sleep(1)
  print(paste0("Downloading ", d, " out of ", length(filesl)))
}