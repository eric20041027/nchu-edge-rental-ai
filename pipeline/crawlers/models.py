"""Shared data models for all rental crawlers."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

CSV_COLUMNS: list[str] = [
    "網址", "地址", "格局", "類型", "室內坪數", "租金",
    "空房間數", "押金", "安全標章", "樓層", "聯絡人", "電話",
    "家具設施", "租金包含", "另計費用", "安全管理", "消防逃生",
    "備註", "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
]


class RentalProperty(BaseModel):
    """A validated rental property record."""
    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(..., alias="網址", min_length=1)
    address: str = Field("", alias="地址")
    layout: str = Field("", alias="格局")
    property_type: str = Field("", alias="類型")
    size: str = Field("", alias="室內坪數")
    rent: str = Field("", alias="租金")
    available_rooms: str = Field("", alias="空房間數")
    deposit: str = Field("", alias="押金")
    safety_cert: str = Field("", alias="安全標章")
    floor: str = Field("", alias="樓層")
    contact_name: str = Field("", alias="聯絡人")
    contact_phone: str = Field("", alias="電話")
    furniture: str = Field("", alias="家具設施")
    rent_includes: str = Field("", alias="租金包含")
    extra_fees: str = Field("", alias="另計費用")
    safety_mgmt: str = Field("", alias="安全管理")
    fire_safety: str = Field("", alias="消防逃生")
    notes: str = Field("", alias="備註")
    image_url: str = Field("", alias="圖片網址")
    distance_km: str = Field("", alias="距離(km)")
    walk_mins: str = Field("", alias="walk_mins")
    scooter_mins: str = Field("", alias="scooter_mins")

    def to_csv_row(self) -> dict[str, str]:
        """Return a dict keyed by Chinese CSV column names."""
        return {
            "網址": self.url,
            "地址": self.address,
            "格局": self.layout,
            "類型": self.property_type,
            "室內坪數": self.size,
            "租金": self.rent,
            "空房間數": self.available_rooms,
            "押金": self.deposit,
            "安全標章": self.safety_cert,
            "樓層": self.floor,
            "聯絡人": self.contact_name,
            "電話": self.contact_phone,
            "家具設施": self.furniture,
            "租金包含": self.rent_includes,
            "另計費用": self.extra_fees,
            "安全管理": self.safety_mgmt,
            "消防逃生": self.fire_safety,
            "備註": self.notes,
            "圖片網址": self.image_url,
            "距離(km)": self.distance_km,
            "walk_mins": self.walk_mins,
            "scooter_mins": self.scooter_mins,
        }
