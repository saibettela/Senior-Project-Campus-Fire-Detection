#ifndef INC_SCD41_H_
#define INC_SCD41_H_

#include "main.h"
#include "sensirion_crc.h"
#include <stdint.h>

#define SCD41_I2C_ADDR                  (0x62 << 1)

/* measurement commands */
#define SCD41_CMD_START_PERIODIC        0x21B1
#define SCD41_CMD_READ_MEASUREMENT      0xEC05
#define SCD41_CMD_STOP_PERIODIC         0x3F86
#define SCD41_CMD_MEASURE_SINGLE        0x219D  /* SCD41 only */
#define SCD41_CMD_DATA_READY            0xE4B8

/* configuration commands */
#define SCD41_CMD_SET_TEMP_OFFSET       0x241D
#define SCD41_CMD_GET_TEMP_OFFSET       0x2318
#define SCD41_CMD_SET_PRESSURE          0xE000
#define SCD41_CMD_FORCED_RECAL          0x362F
#define SCD41_CMD_PERSIST_SETTINGS      0x3615
#define SCD41_CMD_GET_SERIAL            0x3682
#define SCD41_CMD_FACTORY_RESET         0x3632
#define SCD41_CMD_REINIT                0x3646

/* timing (ms) */
#define SCD41_DELAY_STOP_MS             500
#define SCD41_DELAY_SINGLE_SHOT_MS      5000
#define SCD41_DELAY_FORCED_RECAL_MS     400
#define SCD41_DELAY_PERSIST_MS          800
#define SCD41_DELAY_REINIT_MS           20
#define SCD41_DELAY_FACTORY_RESET_MS    1200

typedef struct {
    uint16_t co2_ppm;
    float    temperature_c;
    float    humidity_pct;
} scd41_data_t;

/* periodic measurement */
HAL_StatusTypeDef scd41_start_periodic(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef scd41_stop_periodic(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef scd41_data_ready(I2C_HandleTypeDef *hi2c, uint8_t *ready);
HAL_StatusTypeDef scd41_read_measurement(I2C_HandleTypeDef *hi2c, scd41_data_t *out);

/* single shot (SCD41 only) */
HAL_StatusTypeDef scd41_measure_single(I2C_HandleTypeDef *hi2c, scd41_data_t *out);

/* configuration */
HAL_StatusTypeDef scd41_set_temp_offset(I2C_HandleTypeDef *hi2c, float offset_c);
HAL_StatusTypeDef scd41_get_temp_offset(I2C_HandleTypeDef *hi2c, float *offset_c);
HAL_StatusTypeDef scd41_set_pressure(I2C_HandleTypeDef *hi2c, uint32_t pressure_pa);
HAL_StatusTypeDef scd41_forced_recal(I2C_HandleTypeDef *hi2c, uint16_t target_ppm);
HAL_StatusTypeDef scd41_persist_settings(I2C_HandleTypeDef *hi2c);

/* utility */
HAL_StatusTypeDef scd41_get_serial(I2C_HandleTypeDef *hi2c, uint64_t *serial);
HAL_StatusTypeDef scd41_factory_reset(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef scd41_reinit(I2C_HandleTypeDef *hi2c);

#endif /* INC_SCD41_H_ */
